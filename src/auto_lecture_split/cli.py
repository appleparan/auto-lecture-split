"""CLI module."""

from enum import Enum
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from auto_lecture_split.audio import convert_to_wav, extract_audio
from auto_lecture_split.audio_processing import transcribe_audio, write_transcription
from auto_lecture_split.video import convert_to_mp4
from auto_lecture_split.video_processing import (
    align_transcription_with_slides,
    detect_slide_changes,
)

app = typer.Typer(pretty_exceptions_show_locals=False)
ROOT_DIR = Path(__file__).resolve().parents[2]


class WhisperModelName(str, Enum):
    tiny = 'tiny'
    small = 'small'
    medium = 'medium'
    large = 'large'
    turbo = 'turbo'


class DetectionMethod(str, Enum):
    adaptive = 'adaptive'
    content = 'content'
    threshold = 'threshold'
    histogram = 'histogram'
    hash = 'hash'


@app.command()
def hello() -> None:
    """Hello function for the CLI.

    Prints a simple "Hello, world!" message.
    """
    typer.echo('Hello, world!')


def autocomplete_whisper_model_name() -> list[str]:
    """Auto-completion function for Whisper model names.

    Returns:
        list[str]: A list of available Whisper model names.
    """
    return ['tiny', 'small', 'medium', 'large', 'turbo']


def autocomplete_detection_method() -> list[str]:
    """Auto-completion function for slide change detection methods.

    Returns:
        list[str]: A list of available detection methods.
    """
    return ['adaptive', 'content', 'threshold', 'histogram', 'hash']


@app.command()
def split_video_file(
    video_path: Annotated[str, typer.Argument(help='Path to the input video file.')],
    whipser_model: Annotated[
        WhisperModelName,
        typer.Option(
            help=(
                'Model size to use for transcription. '
                'Available options are '
                '"tiny", "small", "medium", "large", "turbo".'
            ),
            autocompletion=autocomplete_whisper_model_name,
        ),
    ] = WhisperModelName.turbo,
    initial_prompt_path: Annotated[
        str,
        typer.Option(
            help='Path to the initial prompt file.',
        ),
    ] = '',
    language: Annotated[
        str,
        typer.Option(
            help='Language code for the transcription.',
        ),
    ] = 'ko',
    overwrite: Annotated[  # noqa: FBT002
        bool,
        typer.Option(
            help='Whether to overwrite the existing output files.',
        ),
    ] = False,
    detection_method: Annotated[
        DetectionMethod,
        typer.Option(
            help=(
                'Method for slide change detection. '
                'Available options are '
                '"adaptive", "content", "threshold", "histogram", and "hash".'
                'See https://pyscenedetect.readthedocs.io/en/latest/reference/detection-methods/.'
            ),
            autocompletion=autocomplete_detection_method,
        ),
    ] = DetectionMethod.content,
    threshold: Annotated[
        float,
        typer.Option(
            help=(
                'Threshold for slide change detection. Depends on the detection method.'
            )
        ),
    ] = 2.0,
) -> None:
    """Extract, transcribe, and split the video file into slides.

    Args:
        video_path (Annotated[str, typer.Argument):
            Path to the input video file. Defaults to 'Path to the input video file.')].
        whipser_model (Annotated[ WhisperModelName, typer.Option, optional):
             Model size to use for transcription. Defaults to WhisperModelName.turbo
        initial_prompt_path (Annotated[ str, typer.Option, optional):
            Path to the initial prompt file. Defaults to ''.
        language (Annotated[ str, typer.Option, optional):
            Language code for the transcription.. Defaults to 'ko'.
        overwrite (_type_, optional): Whether to overwrite the existing output files.
            Defaults to False.
        detection_method (_type_, optional): Method for slide change detection.
            Available options are "adaptive", "content",
            "threshold", "histogram", and "hash".
            See https://pyscenedetect.readthedocs.io/en/latest/reference/detection-methods/.
            Defaults to DetectionMethod.content.
        threshold (Annotated[ float, typer.Option, optional):
            Threshold for slide change detection. Depends on the detection method.
            Defaults to 2.0.
    """
    # Check if the video file is in MKV format
    video_path = Path(video_path)
    if video_path.suffix.lower() == '.mkv':
        typer.echo('🎬 Converting video file to MP4 format...')
        video_path = convert_to_mp4(video_path, overwrite=overwrite)
        typer.echo('📁 Video file is saved at: ' + str(video_path))

    typer.echo('🎙️ Transcribing audio...')

    file_name = Path(video_path).stem
    audio_path = ROOT_DIR / 'output' / 'audio' / f'{file_name}.wav'
    extract_audio(Path(video_path), Path(audio_path), overwrite=overwrite)
    typer.echo('✅ Audio is save at: ' + str(audio_path))

    # Transcribe the audio
    ## Get initial prompt from the initial_prompt_path
    if initial_prompt_path or initial_prompt_path != '':
        with Path(initial_prompt_path).open('r') as f:
            initial_prompt = f.read()
    transcription_path = (
        ROOT_DIR / 'output' / 'transcription_video_file' / f'{file_name}.vtt'
    )
    transcriptions, trans_res = transcribe_audio(
        audio_path,
        transcription_path=transcription_path,
        initial_prompt=initial_prompt,
        size=whipser_model,
        language=language,
        overwrite=overwrite,
    )
    typer.echo('✅ Transcription completed.')

    if trans_res is not None:
        transcription_path_txt = (
            ROOT_DIR / 'output' / 'transcription_video_file' / f'{file_name}.txt'
        )
        write_transcription(trans_res, transcription_path_txt)
        transcription_path_json = (
            ROOT_DIR / 'output' / 'transcription_video_file' / f'{file_name}.json'
        )
        write_transcription(trans_res, transcription_path_json)

    transcriptions = [
        {'start': seg[0], 'end': seg[1], 'text': str(seg[2])} for seg in transcriptions
    ]

    stat_dir = ROOT_DIR / 'output' / 'stats'
    stat_dir.mkdir(parents=True, exist_ok=True)
    stats_file_path = stat_dir / f'{file_name}_stats.csv'

    slide_changes = detect_slide_changes(
        Path(video_path),
        method=detection_method,
        threshold=threshold,
        stats_file_path=stats_file_path,
    )

    # Print slide change count and details
    typer.echo(
        typer.style(
            f'🔍 Slide changes detected: {len(slide_changes)}',
            fg=typer.colors.BRIGHT_GREEN,
            bold=True,
        )
    )

    # Optionally display the exact timestamps
    for i, timestamp in enumerate(slide_changes, 1):
        msg = (
            f'  ➡ Slide {i}: {timestamp[0].get_timecode()} - '
            f'{timestamp[1].get_timecode()}'
        )
        typer.echo(
            typer.style(
                msg,
                fg=typer.colors.BRIGHT_BLUE,
            )
        )

    typer.echo(
        typer.style(
            '✅ Slide detection completed successfully.',
            fg=typer.colors.BRIGHT_GREEN,
            bold=True,
        )
    )

    df = align_transcription_with_slides(transcriptions, slide_changes)  # noqa: PD901
    typer.echo('✅ Transcription is aligned with slides.')

    output_dir = ROOT_DIR / 'output' / 'final_video_file'
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / f'slides_transcript_{file_name}.csv', index=False)


@app.command()
def transcribe_video_file(
    video_path: Annotated[str, typer.Argument(help='Path to the input video file.')],
    whipser_model: Annotated[
        WhisperModelName,
        typer.Option(
            help=(
                'Model size to use for transcription. '
                'Available options are '
                '"tiny", "small", "medium", "large", "turbo".'
            ),
            autocompletion=autocomplete_whisper_model_name,
        ),
    ] = WhisperModelName.turbo,
    initial_prompt_path: Annotated[
        str,
        typer.Option(
            help='Path to the initial prompt file.',
        ),
    ] = '',
    language: Annotated[
        str,
        typer.Option(
            help='Language code for the transcription.',
        ),
    ] = 'ko',
    overwrite: Annotated[  # noqa: FBT002
        bool,
        typer.Option(
            help='Whether to overwrite the existing output files.',
        ),
    ] = False,
) -> None:
    """Extract, transcribe, and split the video file into slides.

    Args:
        video_path (Annotated[str, typer.Argument):
            Path to the input video file. Defaults to 'Path to the input video file.')].
        whipser_model (Annotated[ WhisperModelName, typer.Option, optional):
             Model size to use for transcription. Defaults to WhisperModelName.turbo
        initial_prompt_path (Annotated[ str, typer.Option, optional):
            Path to the initial prompt file. Defaults to ''.
        language (Annotated[ str, typer.Option, optional):
            Language code for the transcription.. Defaults to 'ko'.
        overwrite (_type_, optional): Whether to overwrite the existing output files.
            Defaults to False.
    """
    # Check if the video file is in MKV format
    video_path = Path(video_path)
    if video_path.suffix.lower() == '.mkv':
        typer.echo('🎬 Converting video file to MP4 format...')
        video_path = convert_to_mp4(video_path, overwrite=overwrite)
        typer.echo('📁 Video file is saved at: ' + str(video_path))

    typer.echo('🎙️ Transcribing audio...')

    file_name = Path(video_path).stem
    audio_path = ROOT_DIR / 'output' / 'audio' / f'{file_name}.wav'
    extract_audio(Path(video_path), Path(audio_path), overwrite=overwrite)
    typer.echo('✅ Audio is save at: ' + str(audio_path))

    # Transcribe the audio
    ## Get initial prompt from the initial_prompt_path
    if initial_prompt_path or initial_prompt_path != '':
        with Path(initial_prompt_path).open('r') as f:
            initial_prompt = f.read()
    transcription_path = (
        ROOT_DIR / 'output' / 'transcription_video_file' / f'{file_name}.vtt'
    )
    transcriptions, trans_res = transcribe_audio(
        audio_path,
        transcription_path=transcription_path,
        initial_prompt=initial_prompt,
        size=whipser_model,
        language=language,
        overwrite=overwrite,
    )
    typer.echo('✅ Transcription completed.')

    if trans_res is not None:
        transcription_path_txt = (
            ROOT_DIR / 'output' / 'transcription_video_file' / f'{file_name}.txt'
        )
        write_transcription(trans_res, transcription_path_txt)
        transcription_path_json = (
            ROOT_DIR / 'output' / 'transcription_video_file' / f'{file_name}.json'
        )
        write_transcription(trans_res, transcription_path_json)

    transcriptions = [
        {'start': seg[0], 'end': seg[1], 'text': str(seg[2])} for seg in transcriptions
    ]


@app.command()
def transcribe_audio_file(
    audio_path: Annotated[str, typer.Argument(help='Path to the input audio file.')],
    whipser_model: Annotated[
        WhisperModelName,
        typer.Option(
            help=(
                'Model size to use for transcription. '
                'Available options are '
                '"tiny", "small", "medium", "large", "turbo".'
            ),
            autocompletion=autocomplete_whisper_model_name,
        ),
    ] = WhisperModelName.turbo,
    initial_prompt_path: Annotated[
        str,
        typer.Option(
            help='Path to the initial prompt file.',
        ),
    ] = '',
    language: Annotated[
        str,
        typer.Option(
            help='Language code for the transcription.',
        ),
    ] = 'ko',
    overwrite: Annotated[  # noqa: FBT002
        bool,
        typer.Option(
            help='Whether to overwrite the existing output files.',
        ),
    ] = False,
) -> None:
    """Transcribe the audio of a video file.

    Args:
        audio_path (Annotated[str, typer.Argument):
            'Path to the input audio file.
        whipser_model (Annotated[ WhisperModelName, typer.Option, optional):
             Model size to use for transcription. Defaults to WhisperModelName.turbo
        initial_prompt_path (Annotated[ str, typer.Option, optional):
            Path to the initial prompt file. Defaults to ''.
        language (Annotated[ str, typer.Option, optional):
            Language code for the transcription.. Defaults to 'ko'.
        overwrite (_type_, optional): Whether to overwrite the existing output files.
            Defaults to False.
    """
    # Convert audio file (mp3 or m4a) to wav format
    audio_path = Path(audio_path)
    file_name = Path(audio_path).stem
    output_audio_path = ROOT_DIR / 'output' / 'audio' / f'{file_name}.wav'

    if audio_path.suffix.lower() in ['.mp3', '.m4a']:
        typer.echo('🎵 Converting audio file to WAV format...')
        audio_path = convert_to_wav(
            Path(audio_path), Path(output_audio_path), overwrite=overwrite
        )
        typer.echo('📁 Audio file is saved at: ' + str(audio_path))
    typer.echo('✅ Using audio at: ' + str(audio_path))

    # Transcribe the audio
    ## Get initial prompt from the initial_prompt_path
    if initial_prompt_path or initial_prompt_path != '':
        with Path(initial_prompt_path).open('r') as f:
            initial_prompt = f.read()
    transcription_path = (
        ROOT_DIR / 'output' / 'transcription_audio_file' / f'{file_name}.vtt'
    )
    transcriptions, trans_res = transcribe_audio(
        audio_path,
        transcription_path=transcription_path,
        initial_prompt=initial_prompt,
        size=whipser_model,
        language=language,
        overwrite=overwrite,
    )
    typer.echo('✅ Transcription completed.')

    if trans_res is not None:
        transcription_path_txt = (
            ROOT_DIR / 'output' / 'transcription_audio_file' / f'{file_name}.txt'
        )
        write_transcription(trans_res, transcription_path_txt)
        transcription_path_json = (
            ROOT_DIR / 'output' / 'transcription_audio_file' / f'{file_name}.json'
        )
        write_transcription(trans_res, transcription_path_json)

    transcriptions = [
        {'start': seg[0], 'end': seg[1], 'text': str(seg[2])} for seg in transcriptions
    ]

    # Convert to DataFrame and strip trailing spaces
    df = pd.DataFrame(transcriptions)  # noqa: PD901
    df['text'] = df['text'].str.strip()

    output_dir = ROOT_DIR / 'output' / 'final_audio_file'
    output_dir.mkdir(parents=True, exist_ok=True)
    # Get filename of video file without extension
    df.to_csv(output_dir / f'transcript_{file_name}.csv', index=False)


def main() -> None:
    """Main function for the CLI."""
    app()
