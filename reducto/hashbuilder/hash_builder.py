from typing import Dict, List, Mapping
import numpy as np
from sklearn import neighbors

from reducto.differencer import DiffProcessor


class ThreshMap:

    def __init__(self, init_dict):
        self.hash_table = init_dict['table']
        self.distri_range = init_dict['distri']
        self.feature_dim = init_dict['dim']
        self.thresh_candidate = init_dict['tcand']

    def get_thresh(self, diff_vector, motion_vector=None):
        diff_vector = np.array(self._histogram(diff_vector))[np.newaxis, :]
        pred_thresh = self.hash_table.predict(diff_vector).item()
        distance, _ = self.hash_table.kneighbors(diff_vector, return_distance=True)
        return self.thresh_candidate[pred_thresh], distance

    def _histogram(self, diff_vector):
        hist, _ = np.histogram(diff_vector, self.feature_dim, range=self.distri_range)
        return hist / len(diff_vector)


class HashBuilder:

    def __init__(self):
        # hyper-parameter
        self.feature_dim = 30
        self.knn_neighbors = 5
        # NOTE n_neighbors must be greater than number of segment
        self.knn_weights = 'distance'

    def generate_threshmap(self,
                           evaluation_results: List[Mapping[DiffProcessor, Mapping[float, Dict]]],
                           diff_vectors: List[Mapping[DiffProcessor, List[float]]],
                           motion_vectors: List[List[float]] = None,
                           target_acc: float = 0.9,
                           safe_zone: float = 0.00) -> Dict:
        # diff_value_range = {
        #     diff_processor: (min_diff_value, max_diff_value)
        # }
        diff_value_range = {}
        for dp_dv in diff_vectors:
            for dp, dv in dp_dv.items():
                if dp not in diff_value_range:
                    diff_value_range[dp] = (min(dv), max(dv))
                else:
                    diff_value_range[dp] = (
                        min([min(dv), diff_value_range[dp][0]]),
                        max([max(dv), diff_value_range[dp][1]])
                    )

        # optimal_thresh = {
        #     diff_processor: [(distri_vector, optimal_thresh)]
        # }
        optimal_thresh = {}
        thresh_candidate = {}
        for dp_er, dp_dv in zip(evaluation_results, diff_vectors):
            for dp, er in dp_er.items():
                dv = dp_dv[dp]
                if dp not in optimal_thresh:
                    optimal_thresh[dp] = []
                if dp not in thresh_candidate:
                    thresh_candidate[dp] = list(er.keys())
                optimal_thresh[dp].append((
                    self._histogram(dv, diff_value_range[dp]),
                    self._get_optimal_thresh(er, target_acc + safe_zone)
                ))

        # hash_table = {
        #     diff_processor: {
        #         'table': knn_model
        #         'distri': range_of_diff_value,
        #         'dim': dimension_of_diff_value_feature
        #     }
        # }
        hash_table = {}
        for dp, segment_state in optimal_thresh.items():
            knn = neighbors.KNeighborsClassifier(self.knn_neighbors, weights=self.knn_weights)
            x = np.array([x[0] for x in segment_state])
            _y = [(thresh_candidate[dp].index(opt) if opt in thresh_candidate[dp] else 0)
                  for opt in [x[1] for x in segment_state]]
            y = np.array(_y)
            knn.fit(x, y)
            hash_table[dp] = {
                'table': knn,
                'distri': diff_value_range[dp],
                'dim': self.feature_dim,
                'tcand': thresh_candidate[dp]
            }
        return hash_table

    def _histogram(self, diff_vector, distri_range):
        hist, _ = np.histogram(diff_vector, self.feature_dim, range=distri_range)
        return hist / len(diff_vector)

    @staticmethod
    def _get_optimal_thresh(er, target_acc):
        optimal_thresh = 0.0
        for thresh, result in er.items():
            thresh = float(thresh)
            result_cross_query = min([abs(x) for x in result.values()])
            if result_cross_query > target_acc and thresh > optimal_thresh:
                optimal_thresh = thresh
        return optimal_thresh
