import bisect
import subprocess
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
import webvtt
import whisper
from rich.progress import Progress
from skimage.metrics import structural_similarity as ssim
from whisper.utils import get_writer


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
    language: str = 'ko',
    initial_prompt: str = '',
    overwrite: bool = False,  # noqa: FBT001, FBT002
) -> list[dict[str, str | int | float]]:
    """Transcribes the given audio file using the Whisper model.

    Args:
        audio_path (Path): Path to the input audio file.
        transcription_path (Path): Path to save the transcription output.
        size (str, optional): Model size to use for transcription.
            Available options are "tiny", "small", "medium", "large", and "turbo".
            Defaults to "turbo".
        language (str, optional): Language code for the transcription.
            Defaults to "ko".
        initial_prompt (str, optional): Path to the initial prompt file.
        overwrite (bool, optional): Overwrite the existing output file.
            Defaults to False.

    Returns:
        list[tuple[str, str, str]:
            A list of transcription segments (start, end, text) in VTT format
    """
    model = whisper.load_model(size)
    if not overwrite and transcription_path.exists():
        # read the existing transcription file
        return [
            (caption.start, caption.end, caption.text)
            for caption in webvtt.read(transcription_path.resolve(), encoding='utf8')
        ]

    result = model.transcribe(
        str(Path(audio_path).resolve()),
        language=language,
        initial_prompt=initial_prompt,
        verbose=False,
    )

    # Create the transcription file directory
    transcription_path.parent.mkdir(parents=True, exist_ok=True)

    with transcription_path.open('w', encoding='utf-8') as file:
        writer = get_writer('vtt', transcription_path.parent)
        writer.write_result(result, file)

    return [
        (caption.start, caption.end, caption.text)
        for caption in webvtt.read(transcription_path.resolve())
    ]


def detect_slide_changes(
    video_path: str, frame_skip: int = 60, threshold: float = 0.5
) -> list[float]:
    """Detects slide changes in the video by analyzing frame differences.

    Args:
        video_path (str): Path to the input video file.
        frame_skip (int, optional): Number of frames to skip between comparisons.
            Defaults to 60.
        threshold (float, optional): Structural similarity threshold to detect changes.
            Defaults to 0.5.

    Returns:
        list[float]: A list of timestamps (in seconds) where slide changes occur.
    """
    cap = cv2.VideoCapture(video_path)
    prev_frame = None
    timestamps = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with Progress() as progress:
        task = progress.add_task(
            '[cyan]Processing video...', total=total_frames // frame_skip
        )
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_frame is not None:
                score = ssim(prev_frame, gray)
                if score < threshold:
                    timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
            prev_frame = gray
            progress.update(task, advance=1)

    cap.release()
    return timestamps


def time_to_seconds(time_str: str) -> float:
    """Convert a timestamp string (HH:MM:SS.sss) to seconds as float."""
    time_obj = datetime.strptime(time_str, '%H:%M:%S.%f')  # noqa: DTZ007
    return (
        time_obj.hour * 3600
        + time_obj.minute * 60
        + time_obj.second
        + time_obj.microsecond / 1e6
    )


def align_transcription_with_slides(
    transcriptions: list[tuple[str, str, str]], slide_times: list[str]
) -> pd.DataFrame:
    """Aligns transcription segments with detected slide changes based on timestamps.

    Args:
        transcriptions (list[tuple[str, str, str]]):
            List of transcription segments with timestamps.
        slide_times (list): List of timestamps indicating slide changes.

    Returns:
        pd.DataFrame: A DataFrame containing start time, end time,
            and transcribed text for each slide.
    """
    # Convert transcription times to float
    transcriptions = [
        (time_to_seconds(seg[0]), time_to_seconds(seg[1]), seg[2])
        for seg in transcriptions
    ]

    # Prepare result storage
    data = [
        {'start': start, 'end': end, 'text': ''}
        for start, end in zip(
            slide_times, slide_times[1:] + [float('inf')], strict=False
        )
    ]

    # Assign transcriptions to slide intervals using bisect
    for start_time, _, text in transcriptions:
        idx = (
            bisect.bisect_right(slide_times, start_time) - 1
        )  # Find the corresponding slide
        if (
            idx >= 0
            and idx < len(data)
            and slide_times[idx] <= start_time < data[idx]['end']
        ):
            data[idx]['text'] += text + ' '

    # Convert to DataFrame and strip trailing spaces
    df = pd.DataFrame(data)  # noqa: PD901
    df['text'] = df['text'].str.strip()

    return df
