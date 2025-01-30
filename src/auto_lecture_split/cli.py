"""CLI module."""

from enum import Enum
from pathlib import Path
from typing import Annotated

import typer

from auto_lecture_split.video_file import (
    align_transcription_with_slides,
    convert_to_mp4,
    detect_slide_changes,
    extract_audio,
    transcribe_audio,
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
def split_file(
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
            help='Overwrite the existing output files.',
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
    """Transcribe the audio of a video file."""
    # Check if the video file is in MKV format
    video_path = Path(video_path)
    if video_path.suffix.lower() == '.mkv':
        typer.echo('🎬 Converting video file to MP4 format...')
        video_path = convert_to_mp4(video_path, overwrite=overwrite)
        typer.echo('📁 Video file is saved at: ' + str(video_path))

    typer.echo('🎙️ Transcribing audio...')

    video_file_name = Path(video_path).stem
    audio_path = ROOT_DIR / 'output' / 'audio' / f'{video_file_name}.wav'
    _ = extract_audio(video_path, audio_path, overwrite=overwrite)
    typer.echo('✅ Audio is save at: ' + str(audio_path))

    # Transcribe the audio
    ## Get initial prompt from the initial_prompt_path
    if initial_prompt_path or initial_prompt_path != '':
        with Path(initial_prompt_path).open('r') as f:
            initial_prompt = f.read()
    transcription_path = (
        ROOT_DIR / 'output' / 'transcription' / f'{video_file_name}.txt'
    )
    transcriptions = transcribe_audio(
        audio_path,
        transcription_path=transcription_path,
        initial_prompt=initial_prompt,
        size=whipser_model,
        language=language,
        overwrite=overwrite,
    )
    typer.echo('✅ Transcription completed.')

    stat_dir = ROOT_DIR / 'output' / 'stats'
    stat_dir.mkdir(parents=True, exist_ok=True)
    video_file_name = Path(video_path).stem
    stats_file_path = stat_dir / f'{video_file_name}_stats.csv'

    slide_changes = detect_slide_changes(
        video_path,
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

    output_dir = ROOT_DIR / 'output' / 'final'
    output_dir.mkdir(parents=True, exist_ok=True)
    # Get filename of video file without extension
    video_file_name = Path(video_path).stem
    df.to_csv(output_dir / f'slides_transcript_{video_file_name}.csv', index=False)


def main() -> None:
    """Main function for the CLI."""
    app()
