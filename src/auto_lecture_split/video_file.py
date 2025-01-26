import subprocess
from pathlib import Path

import cv2
import pandas as pd
import whisper
from skimage.metrics import structural_similarity as ssim


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


def extract_audio(
    video_path: Path,
    audio_path: Path = Path('output') / 'audio.wav',
    overwrite: bool = False,  # noqa: FBT001, FBT002
) -> Path:
    """Extracts and save the audio from the given video file.

    Args:
        video_path (Path): Path to the input video file.
        audio_path (Path, optional): Path to save the extracted audio file.
            Defaults to "output/audio/[video_file_name].wav".
        overwrite (bool, optional): Overwrite the existing output file.
            Defaults to False.

    Returns:
        Path: Path to the saved audio file.
    """
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and audio_path.exists():
        return audio_path
    subprocess.run(  # noqa: S603
        ['ffmpeg', '-i', video_path, '-q:a', '0', '-map', 'a', audio_path, '-y'],  # noqa: S607
        check=True,
    )
    return audio_path


def transcribe_audio(
    audio_path: Path,
    transcription_path: Path,
    size: str = 'turbo',
    overwrite: bool = False,  # noqa: FBT001, FBT002
) -> list[dict[str, str | int | float]]:
    """Transcribes the given audio file using the Whisper model.

    Args:
        audio_path (Path): Path to the input audio file.
        transcription_path (Path): Path to save the transcription output.
        size (str, optional): Model size to use for transcription.
            Available options are "tiny", "small", "medium", "large", and "turbo".
            Defaults to "turbo".
        overwrite (bool, optional): Overwrite the existing output file.
            Defaults to False.

    Returns:
        list[dict[str, str | int | float]]:
            A list of transcription segments containing timestamps and text.
            dictionary keys: 'id', 'seek', 'start', 'end', 'text'. 'temperature',
                'avg_logprob', 'compression_ratio', 'no_speech_prob'
    """
    model = whisper.load_model(size)
    if not overwrite and transcription_path.exists():
        # read the existing transcription file
        with Path(transcription_path).open('r') as file:
            return file.readlines()

    result = model.transcribe(str(Path(audio_path).resolve()))

    with Path(transcription_path).open('w') as file:
        # write the transcription segments entirely to the file
        file.write(''.join(result['segments']))

    return result['segments']


def detect_slide_changes(video_path: str, threshold: float = 0.5) -> list:
    """Detects slide changes in the video by analyzing frame differences.

    Args:
        video_path (str): Path to the input video file.
        threshold (float, optional): Structural similarity threshold to detect changes.
            Defaults to 0.5.

    Returns:
        list: A list of timestamps (in seconds) where slide changes occur.
    """
    cap = cv2.VideoCapture(video_path)
    prev_frame = None
    timestamps = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_frame is not None:
            score = ssim(prev_frame, gray)
            if score < threshold:
                timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
        prev_frame = gray
    cap.release()
    return timestamps


def align_transcription_with_slides(
    transcriptions: list[str], slide_times: list[str]
) -> pd.DataFrame:
    """Aligns transcription segments with detected slide changes based on timestamps.

    Args:
        transcriptions (list): List of transcription segments with timestamps.
        slide_times (list): List of timestamps indicating slide changes.

    Returns:
        pd.DataFrame: A DataFrame containing start time, end time,
            and transcribed text for each slide.
    """
    data = []
    for slide_start, slide_end in zip(
        slide_times, slide_times[1:] + [None], strict=False
    ):
        slide_text = ' '.join(
            seg['text']
            for seg in transcriptions
            if slide_start <= seg['start'] < (slide_end or float('inf'))
        )
        data.append({'start': slide_start, 'end': slide_end, 'text': slide_text})
    return pd.DataFrame(data)
