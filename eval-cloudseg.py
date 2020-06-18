import argparse
import os
import shutil
from collections import namedtuple
from itertools import product
from pathlib import Path
from pprint import pprint

from cloudseg import run_carn
from reducto.codec import img2video
from reducto.data_loader import load_json
from reducto.utils import flatten


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--subset', type=str)
    parser.add_argument('--scale', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--with_bar', action='store_true')
    parser.add_argument('--dataset_root', type=str, default='/mnt/ssd2t/dataset')
    return parser.parse_args()


def get_segments(dataset, subsets, video_list_path=None):
    video_list_path = video_list_path or 'config/video_list.json'
    video_list = load_json(video_list_path)[dataset]
    return [(dataset, i[0], i[1].split('.')[0])
            for i in flatten([list(product([s], video_list[s])) for s in subsets])]


def downsample_video(root, dataset, subset, segment, scale,
                     orig_width=1920, orig_height=1080, batch_size=1,
                     with_bar=False, remove_frames=True):
    scale_str = f'{orig_width // scale}:{orig_height // scale}'
    segment_path = f'{root}/{dataset}/{subset}/{segment}'
    img_path = f'{root}/{dataset}/{subset}/frames/{segment}/x{scale}'
    Path(img_path).mkdir(parents=True, exist_ok=True)
    video_path = f'{root}/{dataset}/{subset}x{scale}/{segment}'

    if Path(video_path).exists():
        return

    # high resolution
    img_path_hr = f'{img_path}/img_%03d_SRF_{scale}_HR.png'
    video2img_hr = f"ffmpeg -hide_banner -loglevel quiet -r 1 -i {segment_path} -r 1 '{img_path_hr}'"
    # print(video2img_hr)
    os.system(video2img_hr)

    # low resolution
    img_path_lr = f'{img_path}/img_%03d_SRF_{scale}_LR.png'
    video2img_lr = f"ffmpeg -hide_banner -loglevel quiet -r 1 -i {segment_path} -r 1 -vf scale={scale_str} '{img_path_lr}'"
    # print(video2img_lr)
    os.system(video2img_lr)

    # super resolution
    sr_root = f'{root}/{dataset}/{subset}x{scale}/frames'
    Path(sr_root).mkdir(parents=True, exist_ok=True)
    carn_args = {
        'model': 'carn',
        'ckpt_path': './checkpoint/carn.pth',
        'sample_dir': sr_root,
        'test_data_dir': f'{root}/{dataset}/{subset}/frames/{segment}',
        'scale': scale,
        'group': 1,
        'shave': 20,
        'cuda': True,
        'batch_size': batch_size,
        'with_bar': with_bar,
        'desc': f'{dataset}/{subset}/{segment}',
    }
    carn_args = namedtuple('args', ' '.join(list(carn_args.keys())))(**carn_args)
    run_carn(carn_args)

    # convert SR images to video
    sr_dir = f'{root}/{dataset}/{subset}x{scale}/frames/{segment}/SR'
    img2video(frame_root=sr_dir, output_path=video_path)

    if remove_frames:
        shutil.rmtree(img_path)
        shutil.rmtree(sr_dir)


if __name__ == '__main__':

    args = parse_args()
    segments = get_segments(args.dataset, [args.subset])
    pprint(args)
    for segment in segments:
        downsample_video(root=args.dataset_root,
                         dataset=args.dataset,
                         subset=args.subset,
                         segment=f'{segment[2]}.mp4',
                         scale=args.scale,
                         batch_size=args.batch_size,
                         with_bar=args.with_bar)
