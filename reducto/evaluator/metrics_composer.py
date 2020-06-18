import logging
import time
from pathlib import Path

from reducto.evaluator import Metrics
from reducto.utils import flatten, assert_list


class MetricComposer:

    def __init__(self, metric_list=None):
        self.metric_list = metric_list
        self.keys = [metric.name for metric in metric_list]

    @staticmethod
    def from_json(config):
        return MetricComposer([
            Metrics.str2class(c['type'])(c['class'])
            for c in config
        ])

    def evaluate_with_diff(self, inference, selected_frames, gt_inference=None, filepath=None):
        results = {}
        for metric in self.metric_list:
            res = metric.evaluate_with_diff(inference, selected_frames, gt_inference)
            results = {**results, **res}
        return results

    def evaluate_single_frame(self, ground_truth, comparision, metric=None):
        results = {}
        if metric:
            metric_list = [m for m in self.metric_list if m.name == metric]
        else:
            metric_list = self.metric_list
        for metric in metric_list:
            res = metric.evaluate_single_frame(ground_truth, comparision)
            results = {
                **results,
                **res,
            }
        return results

    def evaluate_frame_pair(self, pair, inference, metric=None):
        return self.evaluate_single_frame(inference[pair[0]], inference[pair[1]], metric)

    @staticmethod
    def get_frame_pairs(inference, diff_results):
        selected_frame_list = flatten([
            [result['selected_frames'] for _, result in diff_results[key]['result'].items()]
            for key in diff_results.keys()
        ])
        per_frame_list = set(flatten([
            list(zip(inference.keys(), Metrics.interp_frame_ids(frames, len(inference))))
            for frames in selected_frame_list
        ]))
        return per_frame_list

    def evaluate_per_frame(self, inference, diff_results):
        per_frame_list = self.get_frame_pairs(inference, diff_results)
        per_frame_eval = {
            frame_pair: self.evaluate_single_frame(inference[frame_pair[0]], inference[frame_pair[1]])
            for frame_pair in per_frame_list
        }
        return per_frame_eval

    def evaluate(self, inference, diff_results, per_frame_eval=None, filepath=None):
        per_frame_eval = per_frame_eval or self.evaluate_per_frame(inference, diff_results)
        evaluations = {}
        for differ_type, thresh_result in diff_results.items():
            evaluations[differ_type] = {}
            for thresh, diff_result in thresh_result['result'].items():
                selected_frames = diff_result['selected_frames']
                frame_pairs = zip(inference.keys(), Metrics.interp_frame_ids(selected_frames, len(inference)))
                diff_thresh_evaluation = [per_frame_eval[fp] for fp in frame_pairs]
                evaluation = {}
                for key in self.keys:
                    evals = [abs(dte[key]) for dte in diff_thresh_evaluation]
                    evaluation[key] = sum(evals) / len(evals)
                evaluations[differ_type][thresh] = evaluation
        return evaluations

    @staticmethod
    def get(metrics):
        metrics = assert_list(metrics, str)
        metric_list = []
        for metric in metrics:
            if metric == 'mAP-0':
                metric_list.append({'type': 'coco', 'class': [0]})
            if metric == 'mAP-2':
                metric_list.append({'type': 'coco', 'class': [2]})
            if metric == 'counting-0':
                metric_list.append({'type': 'counting', 'class': [0]})
            if metric == 'counting-2':
                metric_list.append({'type': 'counting', 'class': [2]})
            if metric == 'tagging-0':
                metric_list.append({'type': 'tagging', 'class': [0]})
            if metric == 'tagging-2':
                metric_list.append({'type': 'tagging', 'class': [2]})
        return MetricComposer.from_json(metric_list)
