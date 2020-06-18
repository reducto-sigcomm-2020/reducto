import argparse
import importlib
import math
import time
from collections import OrderedDict
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from cloudseg.dataset import TestDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--group', type=int, default=1)
    parser.add_argument('--sample_dir', type=str)
    parser.add_argument('--test_data_dir', type=str)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--shave', type=int, default=20)
    return parser.parse_args()


def save_image(tensor, filename):
    # t1 = time.time()
    tensor = tensor.cpu()
    # t2 = time.time()
    ndarr = tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    # t3 = time.time()
    im = Image.fromarray(ndarr)
    # t4 = time.time()
    im.save(filename)
    # t5 = time.time()
    # print(t2 - t1, t3 - t2, t4 - t3, t5 - t4)


def sample(net, device, dataset, cfg):

    model_name = cfg.ckpt_path.split('.')[0].split('/')[-1]
    sample_dir = Path(cfg.sample_dir)
    data_name = cfg.test_data_dir.split('/')[-1]
    output_dir = sample_dir / model_name / data_name

    sr_dir = output_dir / 'SR'
    sr_dir.mkdir(parents=True, exist_ok=True)

    length = len(dataset)
    batch_size = cfg.batch_size
    loops = math.ceil(len(dataset) / batch_size)

    im_cnt = 0

    if cfg.with_bar:
        pbar = tqdm(total=length, desc=cfg.desc)

    for loop in range(loops):
        lrs = torch.cat([
            dataset[loop * batch_size + i][1].unsqueeze(0)
            for i in range(batch_size)
            if loop * batch_size + i < length
        ]).to(device)
        srs = net(lrs, cfg.scale).detach().squeeze(0)
        if len(srs.shape) == 4:
            for sr in srs:
                sr_im_path = sr_dir / f'{im_cnt:05d}.bmp'
                im_cnt += 1
                save_image(sr, sr_im_path)
                if cfg.with_bar:
                    pbar.update()
        else:
            sr_im_path = sr_dir / f'{im_cnt:05d}.bmp'
            im_cnt += 1
            save_image(srs, sr_im_path)
            if cfg.with_bar:
                pbar.update()

    if cfg.with_bar:
        pbar.close()


def run_carn(cfg):
    module = importlib.import_module(f'cloudseg.model.{cfg.model}')
    net = module.Net(multi_scale=True, group=cfg.group)

    state_dict = torch.load(cfg.ckpt_path)
    new_state_dict = OrderedDict()
    for name, v in state_dict.items():
        new_state_dict[name] = v

    net.load_state_dict(new_state_dict)
    torch.set_grad_enabled(False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)

    dataset = TestDataset(cfg.test_data_dir, cfg.scale)
    sample(net, device, dataset, cfg)
