import logging
import time
from pathlib import Path
import os

from reducto.codec import video2img, img2video, get_video_duration
from reducto.utils import human_readable_size


class Videoer:

    def __init__(self, dataset_root, dataset_name, subset_pattern, master_addr=None, master_port=None):
        self.dataset_root = dataset_root
        self.dataset_name = dataset_name
        self.subset_pattern = subset_pattern
        self.master_addr = master_addr
        self.master_port = master_port
        self.segments = self.generate_segment_list(dataset_root, dataset_name, subset_pattern)
        self.index = 0
        logging.info('Starting transmitting...')
        self.start_time = time.time()

    def send_next(self, diff_processor=None, send_all=False):
        # if there is no more video to be sent, return False
        if self.index >= len(self.segments):
            return False
        # sends video
        filepath, duration, acc_duration = self.segments[self.index]
        video_name = Path(filepath).stem
        # sleep if necessary (to wait for recording video)
        now = time.time()
        eclipsed = now - acc_duration
        if eclipsed < acc_duration:
            time.sleep(int(eclipsed))
        # if there's no need to de-/encode video, send the original video
        if not self.is_codec_required(diff_processor, send_all):
            self.send_video(filepath)
        # otherwise, applying diff and send selected frames as a new encoded video
        else:
            time_start = time.time()
            selected_frames = self.apply_diff(diff_processor, filepath)
            time_end = time.time()
            logging.info(f'{video_name},diff_vector,{diff_processor.__class__.feature},{time_start},{time_end},{time_end - time_start}')

            time_start = time.time()
            generated_video_path = self.select_frames(filepath, selected_frames)
            time_end = time.time()
            logging.info(f'{video_name},diff,{diff_processor.__class__.feature},{time_start},{time_end},{time_end - time_start}')

            self.send_video(generated_video_path)
        # advances iterator and returns True, accounting for the sent video
        self.index += 1
        return True

    @staticmethod
    def generate_segment_list(dataset_root, dataset_name, subset_pattern):
        segment_root = Path(dataset_root) / dataset_name / subset_pattern
        # generates a list of videos of the format [(filepath, duration(filepath))]
        segment_list = [(filepath, get_video_duration(filepath))
                        for filepath in sorted(segment_root.iterdir())
                        if filepath.match('segment???.mp4')]
        # extends the list with accumulated duration
        acc_duration = [sum(map(lambda x: x[1], segment_list[:i+1])) for i in range(len(segment_list))]
        return [(a, b, c) for (a, b), c in zip(segment_list, acc_duration)]

    @staticmethod
    def is_codec_required(diff_processor, send_all):
        if send_all:
            return False
        if diff_processor is None:
            return False
        if diff_processor.thresh != 0:
            return True
        return False

    @staticmethod
    def apply_diff(diff_processor, filepath):
        # logging.info(f'{filepath} start diffing')
        diff_results = diff_processor.process_video(filepath)
        selected_frames = diff_results['selected_frames']
        num_selected_frames = diff_results['num_selected_frames']
        num_total_frames = diff_results['num_total_frames']
        # logging.info(f'{filepath} diff finished ({num_selected_frames}/{num_total_frames})')
        return selected_frames

    @staticmethod
    def select_frames(filepath, selected_frames, codec=False):
        tmp_video_root = Path('/tmp/videoer') / Path(*filepath.parts[-3:])
        tmp_video_path = tmp_video_root / filepath.name
        tmp_frames_folder = tmp_video_root / 'frames'

        if not codec:
            return None

        logging.info(f'{filepath} decoding started')
        video2img(filepath, tmp_frames_folder)
        logging.info(f'{filepath} decoding finished')

        logging.info(f'{filepath} encoding started')
        img2video(tmp_frames_folder, tmp_video_path, selected_frames)
        logging.info(f'{filepath} encoding finished')

        return tmp_video_path

    @staticmethod
    def send_video(filepath):
        # size = os.stat(filepath).st_size
        # readable_size = human_readable_size(size)
        # logging.info(f'{filepath} sent ({readable_size})')
        return
