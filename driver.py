import argparse
import functools
import multiprocessing as mp
from pathlib import Path

import mongoengine
import yaml

from reducto.data_loader import dump_json
from reducto.differencer import DiffComposer
from reducto.evaluator import MetricComposer
from reducto.inferencer import Yolo
from reducto.model import Segment, Inference, InferenceResult, DiffVector, FrameEvaluation

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pipeline', type=str, default='pipeline.yaml')
    parser.add_argument('-s', '--segment_pattern', default='segment???.mp4')
    parser.add_argument('--no_session', action='store_true')
    parser.add_argument('--skip_inference', action='store_true')
    parser.add_argument('--skip_diffeval', action='store_true')
    parser.add_argument('--model', type=str)
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    configuration = args.pipeline
    with open(configuration, 'r') as y:
        config = yaml.load(y, Loader=yaml.FullLoader)

    # data preparation
    dataset_root = config['environs']['dataset_root']
    dataset_name = config['environs']['dataset_name']
    subsets = config['environs']['subsets']
    segments = []
    for ss in subsets:
        p = Path(dataset_root) / dataset_name / ss
        segments += [f for f in sorted(p.iterdir()) if f.match(args.segment_pattern)]

    mongo_host = config['mongo']['host']
    mongo_port = config['mongo']['port']
    mongoengine.connect(dataset_name, host=mongo_host, port=mongo_port)
    print(f'connected to {mongo_host}:{mongo_port} on dataset {dataset_name}')

    differ_dict_path = Path(config['environs']['thresh_root']) / f'{dataset_name}.json'
    differ_types = config['differencer']['types']

    # component preparation
    model = Yolo(no_session=args.no_session)
    differ = DiffComposer.from_jsonfile(differ_dict_path, differ_types)
    evaluator = MetricComposer.from_json(config['evaluator'])

    # pipeline running
    pbar = tqdm(total=len(segments))
    for segment in segments:

        # -- segment ---------------------------------------------------
        segment_record = Segment.find_or_save(segment.parent.name, segment.name)

        # -- inference -------------------------------------------------
        inference_record = Inference.objects(
            segment=segment_record,
            model=model.name,
        ).first()
        if inference_record:
            inference = inference_record.to_json()
        else:
            inference = model.infer_video(segment)
            inference_record = Inference(
                segment=segment_record,
                model=model.name,
                result=[InferenceResult.from_json(inf) for _, inf in inference.items()],
            )
            inference_record.save()
        dump_json(inference, f'data/inference/{dataset_name}/{segment.parent.name}/{segment.stem}.json', mkdir=True)

        # -- skip if required ------------------------------------------
        if args.skip_diffeval:
            pbar.update()
            continue

        # -- diff ------------------------------------------------------
        diff_vectors = {
            dv_record.differencer: dv_record.vector
            for dv_record in DiffVector.objects(segment=segment_record)
        }
        differ_types_pending = [d for d in differ_types if d not in diff_vectors]
        with mp.Pool() as pool:
            dv_f = functools.partial(differ.get_diff_vector, filepath=segment)
            diff_vectors_new = pool.map(dv_f, differ_types_pending)
        diff_vectors_new = {
            differ_type: vector
            for differ_type, vector in zip(differ_types_pending, diff_vectors_new)
        }
        for differ_type, vector in diff_vectors_new.items():
            diff_vector_record = DiffVector(
                segment=segment_record,
                differencer=differ_type,
                vector=vector,
            )
            diff_vector_record.save()

        diff_vectors = {**diff_vectors, **diff_vectors_new}
        diff_results = differ.process_video(segment, diff_vectors)
        dump_json(diff_results, f'data/diff/{dataset_name}/{segment.parent.name}/{segment.stem}.json', mkdir=True)

        # -- evaluation ------------------------------------------------
        frame_pairs = evaluator.get_frame_pairs(inference, diff_results)

        per_frame_evaluations = {}
        for metric in evaluator.keys:
            metric_evaluations = FrameEvaluation.objects(segment=segment_record, evaluator=metric)
            pairs = [(me.ground_truth, me.comparision) for me in metric_evaluations]
            pairs_pending = [p for p in frame_pairs if p not in pairs]
            with mp.Pool() as pool:
                eval_f = functools.partial(evaluator.evaluate_frame_pair, inference=inference, metric=metric)
                metric_evaluations_new = pool.map(eval_f, pairs_pending)
            pair_evaluations_new = {
                pair: evaluation
                for pair, evaluation in zip(pairs_pending, metric_evaluations_new)
            }
            for pair, evaluation in pair_evaluations_new.items():
                frame_evaluation_record = FrameEvaluation(
                    segment=segment_record,
                    model=model.name,
                    evaluator=metric,
                    ground_truth=pair[0],
                    comparision=pair[1],
                    result=evaluation[metric],
                )
                frame_evaluation_record.save()
            for me in metric_evaluations:
                if not per_frame_evaluations.get((me.ground_truth, me.comparision), None):
                    per_frame_evaluations[(me.ground_truth, me.comparision)] = {}
                per_frame_evaluations[(me.ground_truth, me.comparision)][metric] = me.result
            for pair, evaluation in pair_evaluations_new.items():
                if not per_frame_evaluations.get(pair, None):
                    per_frame_evaluations[pair] = {}
                per_frame_evaluations[pair][metric] = evaluation[metric]

        evaluations = evaluator.evaluate(inference, diff_results, per_frame_evaluations, segment)
        dump_json(evaluations, f'data/evaluation/{dataset_name}/{segment.parent.name}/{segment.stem}.json', mkdir=True)

        pbar.update()
