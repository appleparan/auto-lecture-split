import subprocess
from pathlib import Path


def convert_to_wav(
    input_audio_path: Path,
    output_audio_path: Path,
    overwrite: bool = False,  # noqa: FBT001, FBT002
    sample_rate: int = 16000,
) -> Path:
    """Converts an mp3 or m4a audio file to wav format.

    Args:
        input_audio_path (str): Path to the input audio file (mp3 or m4a).
        output_audio_path (str): Path to save the converted audio file.
        overwrite (bool, optional): Overwrite the existing output file.
            Defaults to False.
        sample_rate (int, optional): Sample rate for the output audio file.
            Defaults to 16000.

    Returns:
        str: Path to the converted MP4 video file.
    """
    input_audio_path = Path(input_audio_path)
    if input_audio_path.suffix.lower() in ['.mp3', '.m4a']:
        if not overwrite and output_audio_path.exists():
            return str(output_audio_path)
        cmd = [
            '/usr/bin/ffmpeg',
            '-y',
            '-i',
            str(input_audio_path.resolve()),
            '-ar',
            str(sample_rate),
            '-ac',
            '1',
            '-c:a',
            'pcm_s16le',
            str(output_audio_path.resolve()),
        ]
        subprocess.run(cmd, check=True)  # noqa: S603

        return str(output_audio_path)

    return str(output_audio_path)


def extract_audio(
    input_video_path: Path,
    output_audio_path: Path = Path('output') / 'audio.wav',
    overwrite: bool = False,  # noqa: FBT001, FBT002
) -> Path:
    """Extracts and save the audio from the given video file.

    Args:
        input_video_path (Path): Path to the input video file.
        output_audio_path (Path, optional): Path to save the extracted audio file.
            Defaults to "output/audio/[video_file_name].wav".
        overwrite (bool, optional): Overwrite the existing output file.
            Defaults to False.

    Returns:
        Path: Path to the saved audio file.
    """
    output_audio_path.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and output_audio_path.exists():
        return output_audio_path
    cmd = [
        '/usr/bin/ffmpeg',
        '-i',
        str(input_video_path.resolve()),
        '-q:a',
        '0',
        '-map',
        'a',
        str(output_audio_path.resolve()),
        '-y',
    ]
    subprocess.run(cmd, check=True)  # noqa: S603
    return output_audio_path
