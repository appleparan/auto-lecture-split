from pathlib import Path

import webvtt
import whisper
from whisper.utils import get_writer


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
