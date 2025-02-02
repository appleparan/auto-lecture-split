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
            'ffmpeg',
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
