"""CLI module."""

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


@app.command()
def hello() -> None:
    """Hello function for the CLI.

    Prints a simple "Hello, world!" message.
    """
    typer.echo('Hello, world!')


def whisper_complete_name() -> list[str]:
    """Auto-completion function for Whisper model names.

    Returns:
        list[str]: A list of available Whisper model names.
    """
    return ['tiny', 'small', 'medium', 'large', 'turbo']


@app.command()
def split_file(
    video_path: Annotated[str, typer.Argument(help='Path to the input video file.')],
    whipser_model: Annotated[
        str,
        typer.Option(
            help=(
                'Model size to use for transcription. '
                'Available options are '
                '"tiny", "small", "medium", "large", "turbo".'
            ),
            autocompletion=whisper_complete_name,
        ),
    ] = 'turbo',
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
    frame_skip: Annotated[
        int,
        typer.Option(
            help='Number of frames to skip for slide change detection.',
        ),
    ] = 60,
    threshold: Annotated[
        float,
        typer.Option(
            help='Threshold for slide change detection. (SSIM)',
        ),
    ] = 0.9,
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

    slide_changes = detect_slide_changes(
        video_path, frame_skip=frame_skip, threshold=threshold
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
        typer.echo(
            typer.style(
                f'  ➡ Slide {i}: {timestamp:.2f} seconds', fg=typer.colors.BRIGHT_BLUE
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
