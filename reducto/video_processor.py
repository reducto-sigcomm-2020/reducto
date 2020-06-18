import cv2


class VideoProcessor:

    def __init__(self, video_path, frame_limit=None):
        self.video_path = str(video_path)
        self.frame_limit = frame_limit
        self.frame_count = 0
        self.index = 0
        self.progress_bar = None

    def __enter__(self):
        self.cap = cv2.VideoCapture(self.video_path)
        # self.frame_count = int(cv2.VideoCapture.get(self.cap, int(cv2.CAP_PROP_FRAME_COUNT)))
        self.frame_count = self.cap.get(int(cv2.CAP_PROP_FRAME_COUNT))
        if self.frame_limit and self.frame_limit > 0:
            self.frame_count = min(self.frame_count, self.frame_limit)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cap.release()
        # cv2.destroyAllWindows()

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == self.frame_count:
            raise StopIteration

        _ret, _frame = self.cap.read()
        if not _ret:
            raise StopIteration

        self.index += 1
        if self.progress_bar:
            self.progress_bar.update()
        return _frame

    def __len__(self):
        return self.frame_count
