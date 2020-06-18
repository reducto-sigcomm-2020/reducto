from pathlib import Path

from reducto.codec import get_video_size
from reducto.utils import is_interactive

if is_interactive():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def evaluate_with_selected_frames(dataset, subsets, summary,
                                  metric_type, metric_classes,
                                  mongo_host='thanos.cs.ucla.edu', mongo_port=27017,
                                  model='yolo_v3'):
    import mongoengine
    from reducto.model import Segment, Inference
    from reducto.evaluator import MetricComposer

    evaluator = MetricComposer.from_json([{'type': metric_type, 'class': metric_classes}])
    mongoengine.connect(dataset, host=mongo_host, port=mongo_port)

    segments = get_segments(dataset, subsets)
    evaluations = []
    pbar = tqdm(total=len(segments), desc=f'{dataset}/{subsets}')

    for segment in segments:
        segment_record = Segment.find_or_save(segment[1], f'{segment[2]}.mp4')
        inference = Inference.objects(segment=segment_record, model=model).first().to_json()
        segment_res = [s for s in summary
                       if s['dataset'] == segment[0]
                       and s['subset'] == segment[1]
                       and s['segment'] == segment[2]]
        selected_frames = segment_res[0]['selected_frames']
        evl = evaluator.evaluate_with_diff(inference, selected_frames)
        evaluations.append({
            'dataset': dataset,
            'subset': segment[1],
            'segment': segment[2],
            **evl,
        })
        pbar.update()

    pbar.close()
    mongoengine.disconnect()
    return evaluations


def camera_latency(frame_id, total_frames, fps, divided_by):
    return (1 / fps) * ((total_frames - 1 - frame_id) // divided_by)


def segment_size(dataset, subset, segment, selected_frames=None,
                 video_root=None, log_name=None):
    selected_frames = selected_frames or []
    video_root = video_root or 'dataset'
    video_path = Path(video_root) / dataset / subset / f'{segment}.mp4'

    size = get_video_size(
        video_path, selected_frames=selected_frames if len(selected_frames) > 0 else None,
        remove_output_video=False, log_name=log_name, ugly_fix_for_latency_evaluation=True)
    return size


def frame_latency_focus(summary, bandwidth, rtt, video_root=None, log_name=None,
                        fps=30, segment_duration=5.0, single_inference=1 / 40,
                        with_bar=False, divided_by=1):
    bandwidth_Bps = bandwidth / 8 * 1024 * 1024
    rtt_latency = rtt / 2 / 1000
    report = {'sizes': [], 'latencies': [], 'lat_cam': [], 'lat_net': [], 'lat_inf': []}

    if with_bar:
        pbar = tqdm(total=len(summary))

    for seg in summary:
        num_total_frames = int(fps * segment_duration)
        num_sent_frames = len(seg['selected_frames'])
        size = segment_size(seg['dataset'], seg['subset'], seg['segment'],
                            selected_frames=seg['selected_frames'],
                            video_root=video_root, log_name=log_name)
        report['sizes'].append(size)
        network_latency = (size / bandwidth_Bps + rtt_latency) / float(divided_by)
        selected_frames = seg['selected_frames']

        for frame in range(num_sent_frames):
            cam_latency = camera_latency(selected_frames[frame], num_total_frames, fps, divided_by)
            inf_latency = single_inference * (frame // divided_by + 1)
            latency = cam_latency
            latency += network_latency
            latency += inf_latency
            report['latencies'].append(latency)
            report['lat_cam'].append(cam_latency)
            report['lat_net'].append(network_latency)
            report['lat_inf'].append(inf_latency)

        if with_bar:
            pbar.update()

    if with_bar:
        pbar.close()
    return report


def frame_latency_cloudseg(summary, bandwidth, rtt, video_root=None, log_name=None,
                           fps=30, segment_duration=5.0, single_inference=1 / 30,
                           with_bar=False, divided_by=1):
    bandwidth_Bps = bandwidth / 8 * 1024 * 1024
    rtt_latency = rtt / 2 / 1000
    report = {'sizes': [], 'latencies': [], 'lat_cam': [], 'lat_net': [], 'lat_inf': []}

    if with_bar:
        pbar = tqdm(total=len(summary))

    for segment in summary:
        num_total_frames = int(fps * segment_duration)
        num_sent_frames = num_total_frames
        size = segment_size(segment['dataset'], segment['subset'], segment['segment'],
                            video_root=video_root, log_name=log_name)
        report['sizes'].append(size)
        network_latency = (size / bandwidth_Bps + rtt_latency) / float(divided_by)
        for frame in range(num_sent_frames):
            cam_latency = camera_latency(frame, num_total_frames, fps, divided_by)
            inf_latency = single_inference * (frame // divided_by + 1)
            latency = cam_latency
            latency += network_latency
            latency += inf_latency
            report['latencies'].append(latency)
            report['lat_cam'].append(cam_latency)
            report['lat_net'].append(network_latency)
            report['lat_inf'].append(inf_latency)
        if with_bar:
            pbar.update()

    if with_bar:
        pbar.close()
    return report
