import argparse
from pathlib import Path

from reducto.differencer import PixelDiff, AreaDiff, CornerDiff, EdgeDiff
from reducto.videoer import Videoer


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_name', type=str, default='southampton')
    parser.add_argument('-s', '--subset_pattern', type=str, default='raw000')
    parser.add_argument('-y', '--yes', action='store_true')
    return parser


if __name__ == '__main__':
    args = make_parser().parse_args()
    dataset_name = args.dataset_name
    subset_pattern = args.subset_pattern

    dataset_root = '/mnt/shared/dataset'
    segment_root = Path(dataset_root) / dataset_name / subset_pattern
    segments = [f for f in sorted(segment_root.iterdir()) if f.match('segment???.mp4')]

    videoer = Videoer(dataset_root=dataset_root,
                      dataset_name=dataset_name,
                      subset_pattern=subset_pattern)

    dps = [
        PixelDiff(thresh=0.01),
        AreaDiff(thresh=0.01),
        CornerDiff(thresh=0.01),
        EdgeDiff(thresh=0.01)
    ]

    for dp in dps:
        sent = videoer.send_next(dp)
        while sent is True:
            sent = videoer.send_next(dp)
