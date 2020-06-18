import argparse
import multiprocessing
from functools import partial
from itertools import product
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd

from evaluation import frame_latency_focus, frame_latency_cloudseg
from reducto.evaluator import MetricComposer
from reducto.hashbuilder import HashBuilder, ThreshMap
from reducto.data_loader import load_evaluation, load_diff_vector, load_diff_result, load_json, dump_json
from reducto.utils import flatten, show_stats


actual_size = 2113284

video_root = '/mnt/ssd2t/dataset'
#
# segment_duration = 5.0
# num_bootstrapping_seg = 5
# divided_by = 4
#
# target_acc = 0.90
# tinyyolo_acc = 0.6  # 23.7 / 57.9
# datasets = [
#     {
#         'dataset': 'auburn',
#         'subsets': [
#             'fri005', 'sat001am', 'sat005pm',
#             'fri000', 'fri001', 'fri003', 'fri007', 'fri009', 'fri011',
#             'fri013', 'fri015', 'fri019', 'sat000pm', 'sat001pm', 'sat002pm',
#             'sat006pm'
#         ],
#         'queries': [{
#             'metrics': 'mAP-2', 'differ': 'pixel',
#             'distance': 0.6, 'safe': 0.045, 'target_acc': target_acc,  # reducto
#             'send_thresh': 0.9601, 'tinyyolo_acc': tinyyolo_acc,       # focus
#             'glimpse_best_thresh': 0.0006700584274113686,              # glimpse
#             'glimpse_offline_thresh': 0.0006700584274113686,
#             'glimpse_all_thresh': 0.0006700584274113686,
#             'glimpse_25th_best_thresh': 0.0006700584274113686,
#             'glimpse_25th_offline_thresh': 0.0013401168548227373,
#             'glimpse_25th_all_thresh': 0.0006700584274113686,
#         }],
#         'properties': {'fps': 30, 'segment_duration': segment_duration},
#     },
#     {
#         'dataset': 'jacksonhole',
#         'subsets': [
#             'raw000', 'raw019', 'raw038',
#             'raw001', 'raw002', 'raw003', 'raw004', 'raw005', 'raw006',
#             'raw007', 'raw008', 'raw009', 'raw010', 'raw011', 'raw012',
#             'raw013', 'raw014', 'raw015', 'raw016', 'raw017', 'raw018',
#         ],
#         'queries': [{
#             'metrics': 'mAP-2', 'differ': 'pixel',
#             'distance': 1.9, 'safe': 0.05, 'target_acc': target_acc,
#             'send_thresh': 0.1236, 'tinyyolo_acc': tinyyolo_acc,
#             'glimpse_best_thresh': 0.015458182859056516,
#             'glimpse_offline_thresh': 0.047064646026234565,
#             'glimpse_all_thresh': 0.07701487531565657,
#             'glimpse_25th_best_thresh': 0.016562338777560554,
#             'glimpse_25th_offline_thresh': 0.047064646026234565,
#             'glimpse_25th_all_thresh': 0.04278604184203143,
#         }],
#         'properties': {'fps': 30, 'segment_duration': segment_duration},
#     },
#     {
#         'dataset': 'lagrange',
#         'subsets': [
#             'raw000', 'raw005', 'raw022',
#             'raw001', 'raw002', 'raw003', 'raw004', 'raw006', 'raw007',
#             'raw009', 'raw010', 'raw011', 'raw012', 'raw018', 'raw019',
#             'raw020', 'raw021', 'raw023',
#         ],
#         'queries': [{
#             'metrics': 'mAP-2', 'differ': 'pixel',
#             'distance': 1.8, 'safe': 0.04, 'target_acc': target_acc,
#             'send_thresh': 0.1284, 'tinyyolo_acc': tinyyolo_acc,
#             'glimpse_best_thresh': 0.007982309759177138,
#             'glimpse_offline_thresh': 0.005041458795269771,
#             'glimpse_all_thresh': 0.006301823494087214,
#             'glimpse_25th_best_thresh': 0.006721945060359695,
#             'glimpse_25th_offline_thresh': 0.007142066626632176,
#             'glimpse_25th_all_thresh': 0.006301823494087214,
#         }],
#         'properties': {'fps': 30, 'segment_duration': segment_duration},
#     },
#     {
#         'dataset': 'southampton',
#         'subsets': [
#             'raw001', 'sat005', 'sat009pm',
#             'raw000', 'raw008', 'raw013', 'raw019', 'raw031', 'raw038',
#             'raw042', 'raw043', 'raw049', 'sat000', 'sat000pm', 'sat001pm',
#         ],
#         'queries': [{
#             'metrics': 'mAP-2', 'differ': 'pixel',
#             'distance': 1.5, 'safe': 0.035, 'target_acc': target_acc,
#             'send_thresh': 0.1241, 'tinyyolo_acc': tinyyolo_acc,
#             'glimpse_best_thresh': 0.014329868626743628,
#             'glimpse_offline_thresh': 0.01891542658730159,
#             'glimpse_all_thresh': 0.015476258116883117,
#             'glimpse_25th_best_thresh': 0.008024726430976431,
#             'glimpse_25th_offline_thresh': 0.01662264760702261,
#             'glimpse_25th_all_thresh': 0.013756673881673884,
#         }],
#         'properties': {'fps': 30, 'segment_duration': segment_duration},
#     }
# ]
#
# # networks = [
# #     bandwidth (Mbps), rtt (ms)
# #     {'bandwidth': 12, 'rtt': 100},
# #     {'bandwidth': 24, 'rtt': 20},
# #     {'bandwidth': 60, 'rtt': 5},
# # ]
#
# # network = {'bandwidth': 12, 'rtt': 100}
# # network = {'bandwidth': 24, 'rtt': 20}
# network = {'bandwidth': 60, 'rtt': 5}


def eval_path(seg, root):
    return root / 'evaluation' / seg[0] / seg[1] / f'{seg[2]}.json'


def diff_path(seg, root):
    return root / 'diff' / seg[0] / seg[1] / f'{seg[2]}.json'


def infer_path(seg, root):
    return root / 'inference' / seg[0] / seg[1] / f'{seg[2]}.json'


def get_segments(dataset, subsets, video_list_path=None):
    video_list_path = video_list_path or 'config/video_list.json'
    video_list = load_json(video_list_path)[dataset]
    segments = [(dataset, i[0], i[1].split('.')[0])
                for i in flatten([list(product([s], video_list[s])) for s in subsets])]
    return segments


def driver(f_eval, f_network=None, network_logname=None, heading=''):
    evaluations = {'fractions': [], 'accuracies': [], 'selected_frames': []}
    latencies = {'latencies': [], 'lat_cam': [], 'lat_net': [], 'lat_inf': []}

    with_network = f_network is not None and network_logname is not None

    for video in datasets:
        for query in video['queries']:
            dataset = video['dataset']
            subsets = video['subsets']
            evaluation_summary = f_eval(dataset, subsets, query)
            for item in evaluation_summary:
                evaluations['fractions'].append(item['fraction'])
                evaluations['accuracies'].append(item['evaluation'])
                evaluations['selected_frames'].append(item['selected_frames'])
            if not with_network:
                continue
            network_summary = f_network(
                evaluation_summary, **network, **video['properties'],
                video_root=video_root, log_name=network_logname, divided_by=divided_by)
            # evaluations['sizes'].append(network_summary['sizes'])
            latencies['latencies'].append(network_summary['latencies'])
            latencies['lat_cam'].append(network_summary['lat_cam'])
            latencies['lat_net'].append(network_summary['lat_net'])
            latencies['lat_inf'].append(network_summary['lat_inf'])

    if heading != '':
        print(heading)
        print('-' * 41)
    show_stats(evaluations, ['fractions', 'accuracies'])
    if not with_network:
        print()
        return
    # evaluations['sizes'] = flatten(evaluations['sizes'])
    latencies['latencies'] = flatten(latencies['latencies'])
    latencies['lat_cam'] = flatten(latencies['lat_cam'])
    latencies['lat_net'] = flatten(latencies['lat_net'])
    latencies['lat_inf'] = flatten(latencies['lat_inf'])
    # show_stats(latencies, ['latencies', 'lat_cam', 'lat_net', 'lat_inf'])
    show_stats(latencies, ['latencies'])
    print()


def reducto_eval(dataset, subsets, query, bootstrapping_length=None, with_profiling=True):
    # various time constants
    inference_base_time = 4.5
    profiling_time = 3.5
    camera_diff_time = 2.5
    segment_duration = 5.0
    num_bootstrapping_segment = bootstrapping_length or 5

    segments = get_segments(dataset, subsets)
    metrics = query['metrics']
    differ = query['differ']
    distance_threshold = query['distance']
    safe_zone = query['safe']

    # load data
    profile_root = Path('data')
    evals = [load_evaluation(eval_path(seg, profile_root), differ, metrics) for seg in segments]
    diff_vectors = [load_diff_vector(diff_path(seg, profile_root), differ) for seg in segments]
    diff_results = [load_diff_result(diff_path(seg, profile_root), differ) for seg in segments]
    assert len(evals) == len(diff_vectors) == len(diff_results)
    length = len(evals)

    # start simulating
    profiled_indexes = []
    summary = {}
    for index in range(length):
        summary[index] = {
            'start': index * segment_duration,
            'taken': (index + 1) * segment_duration,
        }
        # bootstrapping phase
        # - camera: sends all segments
        # - server: profiles all segments
        if len(profiled_indexes) < num_bootstrapping_segment:
            profiled_indexes.append(index)
            distance = -1
            fraction = 1.0
            evaluation = 1.0
            summary[index]['sent'] = summary[index]['taken']
            summary[index]['inf_done'] = summary[index]['sent'] + inference_base_time
            summary[index]['prof_done'] = summary[index]['inf_done'] + profiling_time
        # dynamic phase
        # - camera: checks available profiled segments
        # - server: depends on sent segment, either profiles or infers only
        else:
            profiled_available = [
                p_index for p_index in profiled_indexes
                if summary[p_index]['prof_done'] < summary[index]['taken']
            ]
            # if there is not enough profiled segments, still sends everything
            if len(profiled_available) < num_bootstrapping_segment:
                profiled_indexes.append(index)
                # print(diff_vectors[index][differ])
                # print(evals[index][differ])
                distance = -1
                fraction = 1.0
                evaluation = 1.0
                summary[index]['sent'] = summary[index]['taken']
                summary[index]['inf_done'] = summary[index]['sent'] + inference_base_time
                summary[index]['prof_done'] = summary[index]['inf_done'] + profiling_time
            # good to start doing dynamic adoption
            else:
                # camera diffs
                threshmap_init_dict = HashBuilder().generate_threshmap(
                    [evals[i] for i in profiled_indexes],
                    [diff_vectors[i] for i in profiled_indexes],
                    target_acc=target_acc,
                    safe_zone=safe_zone)
                thresh_map = ThreshMap(threshmap_init_dict[differ])
                thresh, distance = thresh_map.get_thresh(diff_vectors[index][differ])
                # print(thresh)
                distance = np.sum(distance)
                if distance > distance_threshold:
                    if with_profiling:
                        profiled_indexes.append(index)
                        # print(diff_vectors[index][differ])
                        # print(evals[index][differ])
                    fraction = 1.0
                    evaluation = 1.0
                    summary[index]['sent'] = summary[index]['taken'] + camera_diff_time
                    summary[index]['inf_done'] = summary[index]['sent'] + inference_base_time
                    summary[index]['prof_done'] = summary[index]['inf_done'] + profiling_time
                else:
                    fraction = diff_results[index][differ][thresh]['fraction']
                    evaluation = evals[index][differ][thresh][metrics]
                    summary[index]['sent'] = summary[index]['taken'] + camera_diff_time
                    summary[index]['inf_done'] = summary[index]['sent'] + inference_base_time * fraction
                    summary[index]['prof_done'] = -1

        summary[index]['profiling?'] = int(index in profiled_indexes)
        summary[index]['diff_vector'] = diff_vectors[index][differ]
        if summary[index]['profiling?']:
            hashmap_evals = evals[index][differ]
            candidates = [he for he in hashmap_evals if hashmap_evals[he][metrics] >= target_acc]
            if len(candidates) == 0:
                summary[index]['hashmap_thresh'] = 0
            else:
                summary[index]['hashmap_thresh'] = max(candidates)

        summary[index]['distance'] = distance
        summary[index]['fraction'] = fraction
        summary[index]['evaluation'] = evaluation

        if fraction == 1.0:
            summary[index]['selected_frames'] = []
        elif fraction < 1.0:
            selected_frames = diff_results[index][differ][thresh]['selected_frames']
            summary[index]['selected_frames'] = selected_frames

        good_threshes = [th for th, acc_dict in evals[index][differ].items()
                         if acc_dict[metrics] > target_acc]
        good_fracs = [(th,
                       diff_results[index][differ][th]['fraction'],
                       evals[index][differ][th][metrics]) for th in good_threshes]
        good_fracs_sorted = sorted(good_fracs, key=lambda th_acc: th_acc[1])

        if len(good_fracs_sorted) == 0:
            optimal_fraction = 1.0
            optimal_evaluation = 1.0
        else:
            optimal_fraction = good_fracs_sorted[0][1]
            optimal_evaluation = good_fracs_sorted[0][2]

        summary[index]['optimal_fraction'] = optimal_fraction
        summary[index]['optimal_evaluation'] = optimal_evaluation
        summary[index]['dataset'] = segments[index][0]
        summary[index]['subset'] = segments[index][1]
        summary[index]['segment'] = segments[index][2]

    summary_list = [summary[v] for v in range(num_bootstrapping_segment, length)]
    return summary_list


def reducto_optimal_eval(dataset, subsets, query):
    segments = get_segments(dataset, subsets)
    metrics = query['metrics']
    differ = query['differ']

    # load data
    profile_root = Path('data')
    evals = [load_evaluation(eval_path(seg, profile_root), differ, metrics) for seg in segments]
    diff_vectors = [load_diff_vector(diff_path(seg, profile_root), differ) for seg in segments]
    diff_results = [load_diff_result(diff_path(seg, profile_root), differ) for seg in segments]
    assert len(evals) == len(diff_vectors) == len(diff_results)
    length = len(evals)

    summary = []
    for index in range(length):

        good_threshes = [th for th, acc_dict in evals[index][differ].items()
                         if acc_dict[metrics] > target_acc]
        good_fracs = [(th,
                       diff_results[index][differ][th]['fraction'],
                       evals[index][differ][th][metrics]) for th in good_threshes]
        good_fracs_sorted = sorted(good_fracs, key=lambda th_acc: th_acc[1])

        if len(good_fracs_sorted) == 0:
            optimal_fraction = 1.0
            optimal_evaluation = 1.0
        else:
            optimal_fraction = good_fracs_sorted[0][1]
            optimal_evaluation = good_fracs_sorted[0][2]

        summary.append({
            'fraction': optimal_fraction,
            'evaluation': optimal_evaluation,
            'dataset': segments[index][0],
            'subset': segments[index][1],
            'segment': segments[index][2],
            'selected_frames': [],
        })

    return summary


def glimpse_eval(dataset, subsets, query, thresh_key='glimpse_oracle_thresh'):

    segments = get_segments(dataset, subsets)
    metrics = query['metrics']
    differ = query['differ']
    glimpse_thresh = query[thresh_key]

    profile_root = Path('data')
    evals = [load_evaluation(eval_path(seg, profile_root), differ, metrics) for seg in segments]
    diff_vectors = [load_diff_vector(diff_path(seg, profile_root), differ) for seg in segments]
    diff_results = [load_diff_result(diff_path(seg, profile_root), differ) for seg in segments]
    assert len(evals) == len(diff_vectors) == len(diff_results)
    length = len(evals)

    summary = []
    for index in range(length):
        summary.append({
            'dataset': segments[index][0],
            'subset': segments[index][1],
            'segment': segments[index][2],
            'fraction': diff_results[index][differ][glimpse_thresh]['fraction'],
            'evaluation': evals[index][differ][glimpse_thresh][metrics],
            'diff_vector': diff_vectors[index][differ],
            'selected_frames': diff_results[index][differ][glimpse_thresh]['selected_frames'],
        })
    return summary


def focus_eval(dataset, subsets, query):
    eval_root = Path('data') / 'focus'
    inference_root = Path('data') / 'inference' / dataset
    send_thresh = query['send_thresh']
    tinyyolo_acc = query['tinyyolo_acc']

    evaluator = MetricComposer.from_json([{'type': 'coco', 'class': [2]}])

    summary = []
    for subset in subsets:
        j = load_json(eval_root / dataset / f'{subset}_car.json')
        for segment_id in sorted(j.keys()):
            if segment_id == '':
                continue
            segment_name = f'segment{segment_id}' if not segment_id.startswith('segment') else segment_id
            frame_ids = sorted(int(fid) for fid in j[segment_id])

            selected_frames = []
            evaluations = []

            for fid in frame_ids:
                inference = j[segment_id][str(fid)]
                scores = inference['scores']
                num_detections = len(inference['scores'])

                if num_detections == 0 or any(float(s) < send_thresh for s in scores):
                    selected_frames.append(fid + 1)
                    evaluations.append(1.0)
                elif tinyyolo_acc > 0.0:
                    evaluations.append(tinyyolo_acc)
                else:
                    detection = {
                        'num_detections': num_detections,
                        'detection_scores': inference['scores'],
                        # [[xmin, ymin, xmax, ymax]]
                        'detection_boxes': [[float(coord) for coord in box] for box in inference['boxes']],
                        'detection_classes': [2] * num_detections,
                    }
                    ground_truth_path = inference_root / subset / f'{segment_name}.json'
                    ground_truth = load_json(ground_truth_path)[str(fid + 1)]
                    ground_truth_filtered = {
                        'num_detections': 0,
                        'detection_scores': [],
                        'detection_boxes': [],
                        'detection_classes': [],
                    }
                    for i in range(ground_truth['num_detections']):
                        if ground_truth['detection_classes'][i] != 2:
                            continue
                        ground_truth_filtered['num_detections'] += 1
                        ground_truth_filtered['detection_scores'].append(ground_truth['detection_scores'][i])
                        ground_truth_filtered['detection_boxes'].append(ground_truth['detection_boxes'][i])
                        ground_truth_filtered['detection_classes'].append(ground_truth['detection_classes'][i])
                    feval = evaluator.evaluate_single_frame(
                        ground_truth=ground_truth_filtered,
                        comparision=detection)['mAP-2']
                    evaluations.append(feval)

            summary.append({
                'dataset': dataset,
                'subset': subset,
                'segment': segment_name,
                'fraction': len(selected_frames) / len(frame_ids),
                'evaluation': sum(evaluations) / len(evaluations),
                'selected_frames': selected_frames,
            })

    return summary


def cloudseg_eval(dataset, subsets, query, scale):
    eval_root = Path('data') / 'cloudseg'
    evals_path = eval_root / f'{dataset}x{scale}.json'
    data = load_json(evals_path)

    summary = []
    for item in data:
        # evaluation didn't evaluate raw001, so we skip it
        if item['subset'] not in subsets:
            continue
        summary.append({
            'dataset': dataset,
            'subset': item['subset'],
            'segment': item['segment'].split('.')[0],
            'evaluation': item['mAP-2'],
            'fraction': 1.0,
            'selected_frames': [],
            'scale': scale,
        })
    return summary


def simple_eval(dataset, subsets, query):
    segments = get_segments(dataset, subsets)
    summary = []
    for seg in segments:
        summary.append({
            'dataset': seg[0],
            'subset': seg[1],
            'segment': seg[2],
            'evaluation': 1.0,
            'fraction': 1.0,
            'selected_frames': [],
        })
    return summary


def reducto_optimizer_acc(dist_safe, _video, _query):
    dataset = _video['dataset']
    subsets = _video['subsets']
    _query['distance'] = dist_safe[0]
    _query['safe'] = dist_safe[1]
    evaluations = {'fractions': [], 'accuracies': []}

    evaluation_summary = reducto_eval(dataset, subsets, _query)
    for item in evaluation_summary:
        evaluations['fractions'].append(item['fraction'])
        evaluations['accuracies'].append(item['evaluation'])

    cuts = [.10, .25, .50, .75, .90]
    df = pd.DataFrame(evaluations)
    return {
        'frac_mean': df['fractions'].mean(),
        'acc_mean': df['accuracies'].mean(),
        'fracs': df.quantile(cuts)['fractions'].to_list(),
        'accs': df.quantile(cuts)['accuracies'].to_list(),
    }


def reducto_optimizer(video):

    distances = [i / 100 for i in list(range(0, 201, 10))]
    safes = [(i - 100) / 1000 for i in list(range(0, 200, 5))]
    dist_safe_prod = list(product(distances, safes))

    query = video['queries'][0]
    with multiprocessing.Pool() as pool:
        result = pool.map(partial(reducto_optimizer_acc, _video=video, _query=query), dist_safe_prod)

    output = [
        {
            'distance': dist_safe_prod[i][0],
            'safe': dist_safe_prod[i][1],
            **result[i]
        }
        for i in range(len(dist_safe_prod))
    ]
    output_path = Path('data') / 'focus' / 'optimization' / f'{video["dataset"]}-{query["metrics"]}-{query["target_acc"]:.2f}.json'
    dump_json(output, output_path, mkdir=True)
    print(f'dumped to {output_path}')
    data_above_acc = [d for d in output if d['accs'][1] > query['target_acc']]
    data_above_acc.sort(key=lambda x: x['frac_mean'])
    pprint(data_above_acc[0])


def focus_optimizer():
    send_threshes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    send_threshes = [i / 10000 for i in range(1240, 1250)]
    # datasets = [datasets[0]]
    for video in datasets:
        for query in video['queries']:
            dataset = video['dataset']
            subsets = video['subsets']
            for send_thresh in send_threshes:
                query['send_thresh'] = send_thresh
                evaluations = {'fractions': [], 'accuracies': [], 'selected_frames': []}
                evaluation_summary = focus_eval(dataset, subsets, query)
                for item in evaluation_summary:
                    evaluations['fractions'].append(item['fraction'])
                    evaluations['accuracies'].append(item['evaluation'])
                    evaluations['selected_frames'].append(item['selected_frames'])
                print(dataset, subsets, send_thresh)
                show_stats(evaluations, ['fractions', 'accuracies'])
                print()


def glimpse_optimizer(video):
    for query in video['queries']:
        evaluations = {'fractions': [], 'accuracies': []}
        dataset = video['dataset']
        # subsets = video['glimpse_subsets'] + video['subsets']
        # subsets = video['glimpse_subsets']
        subsets = video['subsets']
        differ = query['differ']
        threshes = load_json(f'config/threshes/{dataset}.json')[differ][10:]
        for thresh in threshes:
            query['thresh'] = thresh
            evaluation_summary = glimpse_eval(dataset, subsets, query, thresh_key='thresh')
            for item in evaluation_summary:
                evaluations['fractions'].append(item['fraction'])
                evaluations['accuracies'].append(item['evaluation'])
            print(dataset, f'thresh={thresh}')
            show_stats(evaluations, ['accuracies'])
            print()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=int)
    return parser.parse_args()


if __name__ == '__main__':
    datasets = [
        {
            'dataset': 'auburn',
            'subsets': [
                'fri005', 'sat001am', 'sat005pm',
                'fri000', 'fri001', 'fri003', 'fri007', 'fri009', 'fri011',
                'fri013', 'fri015', 'fri019', 'sat000pm', 'sat001pm', 'sat002pm',
                'sat006pm'
            ],
            'queries': [
                {'metrics': 'mAP-0', 'differ': 'edge', 'target_acc': _target_acc}
            ],
            'properties': {'fps': 30},
        },
        {
            'dataset': 'jacksonhole',
            'subsets': [
                'raw000', 'raw019', 'raw038',
                'raw001', 'raw002', 'raw003', 'raw004', 'raw005', 'raw006',
                'raw007', 'raw008', 'raw009', 'raw010', 'raw011', 'raw012',
                'raw013', 'raw014', 'raw015', 'raw016', 'raw017', 'raw018',
            ],
            'queries': [
                {'metrics': 'mAP-0', 'differ': 'edge', 'target_acc': _target_acc}
            ],
            'properties': {'fps': 30},
        },
        {
            'dataset': 'lagrange',
            'subsets': [
                'raw000', 'raw005', 'raw022',
                'raw001', 'raw002', 'raw003', 'raw004', 'raw006', 'raw007',
                'raw009', 'raw010', 'raw011', 'raw012', 'raw018', 'raw019',
                'raw020', 'raw021', 'raw023',
            ],
            'queries': [
                {'metrics': 'mAP-0', 'differ': 'edge', 'target_acc': _target_acc}
            ],
            'properties': {'fps': 30},
        },
        {
            'dataset': 'southampton',
            'subsets': [
                'raw001', 'sat005', 'sat009pm',
                'raw000', 'raw008', 'raw013', 'raw019', 'raw031', 'raw038',
                'raw042', 'raw043', 'raw049', 'sat000', 'sat000pm', 'sat001pm',
            ],
            'queries': [
                {'metrics': 'mAP-0', 'differ': 'edge', 'target_acc': _target_acc}
            ],
            'properties': {'fps': 30},
        },
    ]
    for video in datasets:
        reducto_optimizer(video)
