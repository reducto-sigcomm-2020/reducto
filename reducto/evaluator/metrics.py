import numpy as np

from reducto.evaluator.coco.coco_evaluation import CocoDetectionEvaluator
from reducto.evaluator.coco.label_map_util import create_category_index_from_labelmap
from reducto.utils import redirect


class Metrics:

    def __init__(self, target_classes=None):
        self.target_classes = target_classes
        self.target_str = ':'.join([str(s) for s in target_classes]) if target_classes else 'all'
        self.identifier = 'generic'
        self.name = f'{self.identifier}-{self.target_str}'

    def evaluate_single_frame(self, ground_truth, detection):
        return self({'1': ground_truth}, {'1': detection})

    def evaluate(self):
        results = self._evaluate()
        self._reset()
        return results

    def _load_ground_truth(self, ground_truth_json):
        predictions = Metrics._preprocessing_detection_json(ground_truth_json, self.target_classes)
        for filename, prediction in predictions.items():
            boxes = prediction['detection_boxes']
            classes = prediction['detection_classes']
            if all([c == 0 for c in classes]):
                classes += 1
            scores = prediction['detection_scores']
            self._add_ground_truth(filename, boxes, classes, scores)

    def _load_comparision(self, comparision_json):
        predictions = Metrics._preprocessing_detection_json(comparision_json, self.target_classes)
        for filename, prediction in predictions.items():
            boxes = prediction['detection_boxes']
            classes = prediction['detection_classes']
            if all([c == 0 for c in classes]):
                classes += 1
            scores = prediction['detection_scores']
            self._add_comparision(filename, boxes, classes, scores)

    @staticmethod
    def _preprocessing_detection_json(detection_json, target_classes):
        processed_predictions = {}
        for filename, prediction in detection_json.items():
            if prediction['num_detections'] != 0:
                classes = prediction['detection_classes']
                boxes = prediction['detection_boxes']
                scores = prediction['detection_scores']
                if target_classes:
                    class_filter = [c in target_classes for c in classes]
                    boxes = [b for b, f in list(zip(boxes, class_filter)) if f]
                    scores = [s for s, f in list(zip(scores, class_filter)) if f]
                    classes = [c for c, f in list(zip(classes, class_filter)) if f]
                if len(boxes) == 0:
                    processed_predictions[filename] = {
                        'detection_boxes': np.array([[0, 1, 2, 3]], dtype=np.float32),
                        'detection_classes': np.array([0], dtype=np.uint8),
                        'detection_scores': np.array([0], dtype=np.float32)
                    }
                else:
                    processed_predictions[filename] = {
                        'detection_boxes': np.array(boxes, dtype=np.float32),
                        'detection_classes': np.array(classes, dtype=np.uint8),
                        'detection_scores': np.array(scores, dtype=np.float32),
                    }
            else:
                processed_predictions[filename] = {
                    'detection_boxes': np.array([[0, 1, 2, 3]], dtype=np.float32),
                    'detection_classes': np.array([0], dtype=np.uint8),
                    'detection_scores': np.array([0], dtype=np.float32)
                }
        return processed_predictions

    @staticmethod
    def interp_inference(frame_ids, inference):
        interp_ids = Metrics.interp_frame_ids(frame_ids, len(inference))
        interp_infs = {
            i + 1: inference[fid]
            for (i, fid) in zip(range(len(interp_ids)), interp_ids)
        }
        return interp_infs

    @staticmethod
    def interp_frame_ids(frame_ids, num_frames):
        full_ids, interped_ids = list(range(1, num_frames + 1)), []
        full_index, interp_index, last_number_seen = 0, 0, 0
        while full_index != len(full_ids) and interp_index != len(frame_ids):
            if full_ids[full_index] == frame_ids[interp_index]:
                last_number_seen = frame_ids[interp_index]
                interped_ids.append(last_number_seen)
                full_index += 1
                interp_index += 1
            else:
                interped_ids.append(last_number_seen)
                full_index += 1
        while full_index != len(full_ids):
            interped_ids.append(last_number_seen)
            full_index += 1
        return interped_ids

    def evaluate_with_diff(self, inference, selected_frames, gt_inference=None):
        interp_infs = self.interp_inference(selected_frames, inference)
        results = self(gt_inference or inference, interp_infs)
        return results

    def _evaluate(self):
        raise NotImplementedError()

    def _reset(self):
        raise NotImplementedError()

    def _add_ground_truth(self, filename, boxes, classes, scores):
        raise NotImplementedError()

    def _add_comparision(self, filename, boxes, classes, scores):
        raise NotImplementedError()

    def __call__(self, ground_truth_json, comparision_json):
        self._load_ground_truth(ground_truth_json)
        self._load_comparision(comparision_json)
        return self.evaluate()

    @staticmethod
    def str2class(name):
        return {
            'coco': CocoMetrics,
            'counting': CountingMetrics,
            'tagging': TaggingMetrics,
        }[name]


class CocoMetrics(Metrics):

    def __init__(self, target_classes=None):
        super(CocoMetrics, self).__init__(target_classes)

        self.identifier = 'mAP'
        self.name = f'{self.identifier}-{self.target_str}'
        self.coco_keys = ['mAP']

        label_path = 'config/coco_labels.pbtxt'
        label = create_category_index_from_labelmap(label_path, use_display_name=True)
        label_values = list(label.values())
        self.evaluator = CocoDetectionEvaluator(label_values)

    @redirect(stdout=None, stderr=None)
    def _evaluate(self):
        results = self.evaluator.evaluate()
        results = {
            key.split('/')[-1].replace('.', ''): value
            for key, value in results.items()
        }
        results = {
            f'{key}-{self.target_str}': value
            for key, value in results.items()
            if key in self.coco_keys
        }
        return results

    def _reset(self):
        self.evaluator.clear()

    def _add_ground_truth(self, filename, boxes, classes, scores):
        self.evaluator.add_single_ground_truth_image_info(
            image_id=filename,
            groundtruth_dict={
                'groundtruth_boxes': boxes,
                'groundtruth_classes': classes,
            })

    def _add_comparision(self, filename, boxes, classes, scores):
        self.evaluator.add_single_detected_image_info(
            image_id=filename,
            detections_dict={
                'detection_boxes': boxes,
                'detection_scores': scores,
                'detection_classes': classes,
            })


class CountingMetrics(Metrics):

    def __init__(self, target_classes=None):
        super(CountingMetrics, self).__init__(target_classes)

        self.identifier = 'counting'
        self.name = f'{self.identifier}-{self.target_str}'

        self.ground_truth = {}
        self.comparision = {}
        self.ground_truth_frames = set()
        self.comparision_frames = set()

    def _evaluate(self):
        assert self.ground_truth_frames == self.comparision_frames
        score_list = []
        for frame_id in self.ground_truth_frames:
            gt_found = self.ground_truth[frame_id]
            cmp_found = self.comparision[frame_id]
            if max(gt_found, cmp_found) == 0 or gt_found == cmp_found:
                score_list.append(1)
            else:
                difference = abs(gt_found - cmp_found)
                base = max(gt_found, cmp_found)
                score = (base - difference) / base
                # score = abs(gt_found - cmp_found) / max(gt_found, cmp_found)
                score_list.append(score)
                # print(f'gt: {gt_found}, cmp: {cmp_found}, score: {score}, '
                #       f'abs(gt-cmp): {abs(gt_found - cmp_found)}, '
                #       f'max(gt,cmp): {max(gt_found, cmp_found)}')
        results = {
            self.name: sum(score_list) / len(score_list)
        }
        return results

    def _reset(self):
        self.ground_truth = {}
        self.comparision = {}
        self.ground_truth_frames = set()
        self.comparision_frames = set()

    def _add_ground_truth(self, filename, boxes, classes, scores):
        self.ground_truth_frames.add(filename)
        items_found = sum([int((classes == i).sum()) for i in self.target_classes])
        self.ground_truth[filename] = items_found

    def _add_comparision(self, filename, boxes, classes, scores):
        self.comparision_frames.add(filename)
        items_found = sum([int((classes == i).sum()) for i in self.target_classes])
        self.comparision[filename] = items_found


class TaggingMetrics(Metrics):

    def __init__(self, target_classes=None):
        super(TaggingMetrics, self).__init__(target_classes)

        self.identifier = 'tagging'
        self.name = f'{self.identifier}-{self.target_str}'

        self.ground_truth = {}
        self.comparision = {}
        self.ground_truth_frames = set()
        self.comparision_frames = set()

    def _evaluate(self):
        assert self.ground_truth_frames == self.comparision_frames
        score_list = []
        for frame_id in self.ground_truth_frames:
            gt_found = 1 if self.ground_truth[frame_id] > 0 else 0
            cmp_found = 1 if self.comparision[frame_id] > 0 else 0
            score_list.append(1 - (gt_found ^ cmp_found))
        results = {
            self.name: sum(score_list) / len(score_list)
        }
        return results

    def _reset(self):
        self.ground_truth = {}
        self.comparision = {}
        self.ground_truth_frames = set()
        self.comparision_frames = set()

    def _add_ground_truth(self, filename, boxes, classes, scores):
        self.ground_truth_frames.add(filename)
        items_found = sum([int((classes == i).sum()) for i in self.target_classes])
        self.ground_truth[filename] = items_found

    def _add_comparision(self, filename, boxes, classes, scores):
        self.comparision_frames.add(filename)
        items_found = sum([int((classes == i).sum()) for i in self.target_classes])
        self.comparision[filename] = items_found
