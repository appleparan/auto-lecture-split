import bisect
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd
import scenedetect
import webvtt
import whisper
from scenedetect.detectors.adaptive_detector import AdaptiveDetector
from scenedetect.detectors.content_detector import ContentDetector
from scenedetect.detectors.hash_detector import HashDetector
from scenedetect.detectors.histogram_detector import HistogramDetector
from scenedetect.detectors.threshold_detector import ThresholdDetector
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
    video_path: str,
    method: str,
    threshold: float = 10.0,
    stats_file_path: Path = Path('output') / 'stats.csv',
) -> list[tuple[scenedetect.FrameTimecode, scenedetect.FrameTimecode]]:
    """Detects slide changes in the video by scenedetector.

    Args:
        video_path (str): Path to the input video file.
        method (str): Method for slide change detection.
        threshold (float, optional): Structural similarity threshold to detect changes.
            Defaults to 10.0.
        stats_file_path (Path, optional): Path to save the detection statistics.

    Returns:
        list[tuple[scenedetect.FrameTimecode, scenedetect.FrameTimecode]]:
            List of scenes as pairs of (start, end) FrameTimecode objects.
    """
    stats_manager = scenedetect.StatsManager()
    scene_manager = scenedetect.SceneManager(stats_manager=stats_manager)
    video = scenedetect.open_video(str(video_path.resolve()))

    if method == 'adaptive':
        detector = AdaptiveDetector(adaptive_threshold=threshold)
    elif method == 'content':
        detector = ContentDetector(threshold=threshold)
    elif method == 'threshold':
        detector = ThresholdDetector(threshold=threshold)
    elif method == 'histogram':
        detector = HistogramDetector(threshold=threshold)
    elif method == 'hash':
        detector = HashDetector(threshold=threshold)
    else:
        msg = f'Invalid detection method: {method}'
        raise ValueError(msg)

    scene_manager.add_detector(detector)

    # Detect all scenes in video from current position to end.
    scene_manager.detect_scenes(video, show_progress=True)
    timestamps: list[tuple[scenedetect.FrameTimecode, scenedetect.FrameTimecode]] = (
        scene_manager.get_scene_list()
    )

    # Save per-frame statistics to disk.
    scene_manager.stats_manager.save_to_csv(csv_file=str(stats_file_path.resolve()))

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
    transcriptions: list[tuple[str, str, str]],
    slide_times: list[tuple[scenedetect.FrameTimecode, scenedetect.FrameTimecode]],
) -> pd.DataFrame:
    """Aligns transcription segments with detected slide changes based on timestamps.

    Args:
        transcriptions (list[tuple[str, str, str]]):
            List of transcription segments with timestamps.
        slide_times (list[tuple[scenedetect.FrameTimecode, scenedetect.FrameTimecode]]):
            List of timestamps indicating slide changes.

    Returns:
        pd.DataFrame: A DataFrame containing start time, end time,
            and transcribed text for each slide.
    """
    # Convert transcription times to float
    transcriptions = [
        (time_to_seconds(seg[0]), time_to_seconds(seg[1]), seg[2])
        for seg in transcriptions
    ]

    # Convert slide times to float (seconds)
    slide_times_float = [start.get_seconds() for start, _ in slide_times]

    # Prepare result storage
    data = [
        {'start': start.get_seconds(), 'end': end.get_seconds(), 'text': ''}
        for start, end in slide_times
    ]

    # Assign transcriptions to slide intervals using bisect
    for start_time, _, text in transcriptions:
        idx = (
            bisect.bisect_right(slide_times_float, start_time) - 1
        )  # Find the corresponding slide
        if (
            idx >= 0
            and idx < len(data)
            and slide_times_float[idx] <= start_time < data[idx]['end']
        ):
            data[idx]['text'] += text + ' '

    # Convert to DataFrame and strip trailing spaces
    df = pd.DataFrame(data)  # noqa: PD901
    df['text'] = df['text'].str.strip()

    return df
