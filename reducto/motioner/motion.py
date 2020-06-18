import logging
import time
from pathlib import Path

import cv2
import pybgs

from reducto.utils import redirect


class MotionExtractor:

    def __init__(self, reset=True):
        self.bgs = None
        self.reset = reset
        self.name = '__generic_motion_extractor__'

    def extract_motion(self, video_path):
        time_start = time.time()
        self._reset()
        motion_ratio = []
        cap = cv2.VideoCapture(str(video_path))
        index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                cap.release()
                break
            index += 1
            motion_ratio.append(self.cal_motion_ratio(frame))
        time_end = time.time()
        video_name = Path(video_path).stem
        logging.info(f'{video_name},motion,{self.name},{time_start},{time_end},{time_end - time_start}')
        return motion_ratio

    def cal_motion_ratio(self, frame):
        fgmask = self.bgs.apply(frame)
        motion_ratio = cv2.countNonZero(fgmask) / (frame.shape[0] * frame.shape[1])
        return motion_ratio

    def _reset(self):
        raise NotImplementedError()

    @staticmethod
    def from_json(config):
        motioner_dict = {
            'adaptive_bg_learning': AdaptiveBackgroundLearning,
            'weighted_moving_mean': WeightedMovingMean,
            'sigma_delta': SigmaDelta,
        }
        if config['type'] in motioner_dict:
            return motioner_dict[config['type']]()
        else:
            return None


class AdaptiveBackgroundLearning(MotionExtractor):

    @redirect(stdout=None, stderr=None)
    def __init__(self, reset=True):
        super(AdaptiveBackgroundLearning, self).__init__(reset)
        self.bgs = pybgs.AdaptiveBackgroundLearning()
        self.name = 'adaptive_bg_learning'

    @redirect(stdout=None, stderr=None)
    def _reset(self):
        self.bgs = pybgs.AdaptiveBackgroundLearning()


class WeightedMovingMean(MotionExtractor):

    @redirect(stdout=None, stderr=None)
    def __init__(self, reset=True):
        super(WeightedMovingMean, self).__init__(reset)
        self.bgs = pybgs.WeightedMovingMean()
        self.name = 'weighted_moving_mean'

    @redirect(stdout=None, stderr=None)
    def _reset(self):
        self.bgs = pybgs.WeightedMovingMean()


class SigmaDelta(MotionExtractor):

    @redirect(stdout=None, stderr=None)
    def __init__(self, reset=True):
        super(SigmaDelta, self).__init__(reset)
        self.bgs = pybgs.SigmaDelta()
        self.name = 'sigma_delta'

    @redirect(stdout=None, stderr=None)
    def _reset(self):
        self.bgs = pybgs.SigmaDelta()
