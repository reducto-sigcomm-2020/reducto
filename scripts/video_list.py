from pathlib import Path

from data_loader import dump_json

if __name__ == '__main__':
    dataset_root = '/mnt/shared/dataset'
    names = [
        'auburn', 'banff', 'broadcourt', 'clintonthomas', 'gebhardt',
        'intersection', 'jacksonhole', 'lagrange', 'southampton',
    ]
    video_list = {
        name: {
            subset.name: [
                segment.name
                for segment
                in sorted((Path(dataset_root) / name / subset).iterdir())
                if segment.match('segment???.mp4')]
            for subset in [
                s
                for s
                in sorted((Path(dataset_root) / name).iterdir())
                if s.is_dir()
            ]
        }
        for name in names
    }
    dump_json(video_list, 'video_list.json')
