from pathlib import Path
from reducto.data_loader import load_json
from reducto.utils import flatten
import numpy as np


def load_diff(dataset, differ, metric, target_acc):
    diff_path = Path('data') / 'diff' / dataset
    eval_root = Path('data') / 'evaluation' / dataset
    subsets = [s for s in sorted(diff_path.iterdir())
               if not s.match('*x2') and not s.match('*x3') and not s.match('*x4')]
    segments = flatten([sorted(s.iterdir()) for s in subsets])

    summary = []
    for seg in segments:
        subset = seg.parent.stem
        segment = seg.stem
        eval_path = eval_root / subset / f'{segment}.json'
        evl = load_json(eval_path)[differ]
        diff = load_json(seg)[differ]['result']
        segment_summary = []
        for thresh in diff:
            if evl[thresh][metric] < target_acc:
                continue
            segment_summary.append({
                'dataset': dataset,
                'subset': subset,
                'segment': segment,
                'threshold': thresh,
                'differ': differ,
                'fraction': diff[thresh]['fraction'],
                'evaluation': evl[thresh][metric],
            })
        sorted_segment_summary = sorted(segment_summary, key=lambda x: x['fraction'])
        if len(sorted_segment_summary) > 0:
            summary.append(sorted_segment_summary[0])
    return summary


if __name__ == '__main__':
    # metrics = ['tagging-2', 'tagging-0', 'mAP-2', 'mAP-0', 'counting-2', 'counting-0']
    metrics = ['counting-2', 'mAP-2']
    datasets = ['jacksonhole', 'lagrange', 'southampton', 'auburn', 'clintonthomas']
    differs = ['pixel', 'area', 'edge']

    for metric in metrics:
        print(f'{metric},pixel,area,edge')
        for dataset in datasets:
            print(dataset, end=',')
            for differ in differs:
                summary = load_diff(dataset, differ, metric, 0.70)
                fraction_mean = np.mean([item['fraction'] for item in summary])
                print(f'{fraction_mean:.4f}', end=',')
            print()
        print()
