from pathlib import Path

from data_loader import load_json
from reducto.evaluator import MetricComposer


def evaluate_frame_pair(dataset1, subset1, segment1, fid1,
                        dataset2, subset2, segment2, fid2,
                        metrics):
    inf1 = load_frame_inference(dataset1, subset1, segment1, fid1)
    inf2 = load_frame_inference(dataset2, subset2, segment2, fid2)
    evaluator = MetricComposer.get(metrics)
    results = evaluator.evaluate_single_frame(inf1, inf2)
    return results


def load_frame_inference(dataset, subset, segment, fid):
    root = Path('data')
    inference_path = root / 'inference' / dataset / subset / f'{segment}.json'
    inference = load_json(inference_path)
    return inference[str(fid)]


if __name__ == '__main__':
    # example
    evaluate_frame_pair('auburn', 'fri005', 'segment000', 1,
                        'auburn', 'fri005', 'segment000', 3,
                        ['mAP-2', 'mAP-0', 'tagging-0', 'tagging-2'])
