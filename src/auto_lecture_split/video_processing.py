import html
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import scenedetect
from scenedetect.detectors.adaptive_detector import AdaptiveDetector
from scenedetect.detectors.content_detector import ContentDetector
from scenedetect.detectors.hash_detector import HashDetector
from scenedetect.detectors.histogram_detector import HistogramDetector
from scenedetect.detectors.threshold_detector import ThresholdDetector

# Regex to remove HTML tags
PAT_HTML_TAGS = re.compile(r'</?[^>]+>')


def clean_text(txt: str) -> str:
    """Remove HTML tags (e.g., <c>, <b>) and decode HTML entities."""
    txt = PAT_HTML_TAGS.sub('', txt)  # Remove all HTML tags
    txt = html.unescape(txt)  # Unescape HTML entity(i.e. &nbsp;)
    return txt.strip()


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
    transcriptions = [(seg['start'], seg['end'], seg['text']) for seg in transcriptions]

    # Convert transcription times to DataFrame with float timestamps
    transcript_df = pd.DataFrame(transcriptions, columns=['start', 'end', 'text'])
    transcript_df['start'] = transcript_df['start'].apply(time_to_seconds)
    transcript_df['end'] = transcript_df['end'].apply(time_to_seconds)
    # Remove HTML code
    transcript_df['text'] = transcript_df['text'].apply(clean_text)

    # Convert slide times to DataFrame with float timestamps
    slide_df = pd.DataFrame(
        {
            'start': [start.get_seconds() for start, _ in slide_times],
            'end': [end.get_seconds() for _, end in slide_times],
            'start_time': [start.get_timecode() for start, _ in slide_times],
            'end_time': [end.get_timecode() for _, end in slide_times],
            'text': [''] * len(slide_times),
        }
    )

    # Assign transcriptions to slide intervals using `.between()`
    for idx, slide in slide_df.iterrows():
        matching_transcripts = transcript_df[
            transcript_df['start'].between(slide['start'], slide['end'])
        ]
        slide_df.loc[idx, 'text'] = ' '.join(matching_transcripts['text'].tolist())

    # Strip any trailing spaces in transcript text
    slide_df['text'] = slide_df['text'].str.strip()

    return slide_df
