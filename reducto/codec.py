import os
import shutil
from pathlib import Path
from subprocess import check_output, CalledProcessError, STDOUT


def video2img(video_path, frame_root, extension='bmp', scale=1):
    orig_width = 1920
    orig_height = 1080
    scale_str = f'{orig_width // scale}:{orig_height // scale}'
    frame_root.mkdir(parents=True, exist_ok=True)
    # ffmpeg -r 1 -i segment000.mp4 -r 1 "/tmp/frames/%05d.bmp"
    command = f'ffmpeg -hide_banner -loglevel quiet -r 1 -i {video_path} -r 1 -vf scale={scale_str} "{frame_root}/%05d.{extension}"'
    os.system(command)
    frames = [f for f in sorted(frame_root.iterdir()) if f.match(f'?????.{extension}')]
    return len(frames)


def img2video(frame_root, output_path, selected_frames=None,
              frame_pattern='?????', extension='bmp'):
    # if output_path.exists():
    #     return
    frame_root = Path(frame_root)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if selected_frames is None:
        selected_frames = [f.stem for f in sorted(frame_root.iterdir()) if f.match(f'{frame_pattern}.{extension}')]
    # print(f'img2video {frame_root} ({len(selected_frames)}) ... ', end='')
    frame_list = [f'{frame_root}/{int(i):05d}.{extension}' for i in selected_frames]
    frame_str = ' '.join(frame_list)
    command = f'cat {frame_str} | ' \
              f'ffmpeg -hide_banner -loglevel panic ' \
              f'-f image2pipe -framerate 30 -i - {output_path}'
    os.system(command)
    # print('done')


def get_video_size(input_video_path, selected_frames=None, log_name=None,
                   remove_output_video=False, scale=1):
    """
    Returns canonical video size. If `dp_name` and `selected_frames` are both
    provided, returns diffed video size; otherwise, returns original video size.
    """
    input_video_path = Path(input_video_path)
    video_name = input_video_path.stem

    diff_video_root = input_video_path.parent / 'diff_videos'
    if not diff_video_root.parent.exists():
        diff_video_root.parent.mkdir(parents=True, exist_ok=True)

    if log_name:
        output_video_path = diff_video_root / log_name / f'{video_name}.mp4'
    else:
        output_video_path = diff_video_root / f'{video_name}.mp4'

    if not output_video_path.exists():
        frame_root = input_video_path.parent / 'frames' / video_name
        video2img(input_video_path, frame_root, scale=scale)
        img2video(frame_root, output_video_path, selected_frames)
        shutil.rmtree(frame_root)

    size = os.stat(output_video_path).st_size
    if remove_output_video:
        os.remove(output_video_path)
    return size


def get_video_duration(filepath):
    cmd = ['ffprobe',
           '-v',
           'error',
           '-show_entries',
           'format=duration',
           '-of',
           'default=noprint_wrappers=1:nokey=1',
           filepath]
    try:
        output = check_output(cmd, stderr=STDOUT).decode()
    except CalledProcessError as ex:
        output = ex.output.decode()
    return int(float(output))
