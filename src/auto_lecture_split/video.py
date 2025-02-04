import subprocess
from pathlib import Path


def convert_to_mp4(video_path: Path, overwrite: bool = False) -> Path:  # noqa: FBT001, FBT002
    """Converts an MKV video file to MP4 format.

    Args:
        video_path (str): Path to the input MKV video file.
        overwrite (bool, optional): Overwrite the existing output file.
            Defaults to False.

    Returns:
        str: Path to the converted MP4 video file.
    """
    video_path = Path(video_path)
    if video_path.suffix.lower() == '.mkv':
        mp4_path = video_path.with_suffix('.mp4')
        if not overwrite and mp4_path.exists():
            return str(mp4_path)
        subprocess.run(  # noqa: S603
            [  # noqa: S607
                'ffmpeg',
                '-i',
                str(video_path),
                '-c:v',
                'copy',
                '-c:a',
                'aac',
                str(mp4_path),
                '-y',
            ],
            check=True,
        )
        return str(mp4_path)
    return str(video_path)
