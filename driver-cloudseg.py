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

from pprint import pprint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pipeline', type=str, default='pipeline.yaml')
    parser.add_argument('-s', '--segment_pattern', default='segment???.mp4')
    parser.add_argument('--no_session', action='store_true')
    parser.add_argument('--skip_inference', action='store_true')
    parser.add_argument('--skip_diffeval', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    configuration = args.pipeline
    with open(configuration, 'r') as y:
        config = yaml.load(y, Loader=yaml.FullLoader)

    dataset_root = config['environs']['dataset_root']
    mongo_host = config['mongo']['host']
    mongo_port = config['mongo']['port']
    datasets = config['datasets']

    model = Yolo(no_session=args.no_session)
    evaluator = MetricComposer.from_json(config['evaluator'])

    for video in datasets:

        dataset = video['dataset']
        subsets = video['subsets']
        scale = video['scale']
        subsets_scaled = [f'{s}x{scale}' for s in subsets]

        mongoengine.connect(dataset, host=mongo_host, port=mongo_port)

        segments = []
        for ss in subsets:
            p = Path(dataset_root) / dataset / ss
            segments += [f for f in sorted(p.iterdir()) if f.match(args.segment_pattern)]

        segments_scaled = []
        for ss in subsets_scaled:
            p = Path(dataset_root) / dataset / ss
            segments_scaled += [f for f in sorted(p.iterdir()) if f.match(args.segment_pattern)]

        assert len(segments) == len(segments_scaled)

        print(f'{"-" * 32}')
        print(f'mongo: {mongo_host}:{mongo_port}')
        print(f'dataset: {dataset}')
        print(f'subsets: {subsets}, {len(segments)} videos')
        print(f'subsets_scaled: {subsets_scaled}, {len(segments_scaled)} videos')
        print(f'{"-" * 32}')

        pbar = tqdm(total=len(segments), desc=f'{dataset} {scale}x')

        result_dir = Path('data') / 'cloudseg'
        result_dir.mkdir(parents=True, exist_ok=True)
        # result_csv_path = result_dir / f'{dataset}x{scale}.csv'
        # result_csv = open(result_csv_path, 'w')
        # result_csv.write(f'subset,segment,' + ','.join(evaluator.keys()))

        result_json_path = result_dir / f'{dataset}x{scale}.json'
        result = []

        for seg, seg_scaled in zip(segments, segments_scaled):

            assert f'{seg.parent.name}x{scale}' == seg_scaled.parent.name
            assert seg.name == seg_scaled.name

            seg_record = Segment.find_or_save(seg.parent.name, seg.name)
            seg_scaled_record = Segment.find_or_save(seg_scaled.parent.name, seg_scaled.name)

            seg_inf = Inference.objects(segment=seg_record, model=model.name).first().to_json()
            seg_scaled_inf = Inference.objects(segment=seg_scaled_record, model=model.name).first().to_json()

            if len(seg_inf.keys()) != len(seg_scaled_inf.keys()):
                continue

            frame_evals = [
                evaluator.evaluate_single_frame(seg_inf[fid], seg_scaled_inf[fid])
                for fid in list(seg_inf.keys())
            ]
            # only valid for mAP metrics
            avg_evals = {
                metric: sum(evl[metric] for evl in frame_evals) / len(frame_evals)
                for metric in evaluator.keys
            }
            result.append({
                'subset': seg.parent.name,
                'segment': seg.name,
                'scale': scale,
                **avg_evals,
            })

            eval_str = f'{seg.parent.name}/{seg.stem} ' + ','.join(f'{k}={v:.4f}' for k, v in avg_evals.items())
            pbar.set_postfix_str(eval_str)

            pbar.update()

        dump_json(result, result_json_path)
        pbar.close()
        mongoengine.disconnect()
