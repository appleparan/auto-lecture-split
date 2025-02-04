# auto-lecture-split

Inspired by [hyesik/auto-lecture-note](https://github.com/hyeshik/auto-lecture-note),
this tool processes lecture videos and generates structured transcripts with precise slide-based segmentation.

## ✨ Features
* 📼 Lecture Video/Audio Processing – Supports MKV/MP4 and M4A/MP3 files
* 📝 Whisper-Based Transcription – Leverages OpenAI's Whisper for accurate speech-to-text conversion
* 📊 Slide Transition Detection – Automatically segments text based on slide changes
* ⏳ Timestamped Notes – Creates structured lecture notes aligned with the timeline

📌 How to Use

1. Place your file (video/audio)
    * Video: Place your `<VIDEO_FILE_NAME>.mkv` or `<VIDEO_FILE_NAME>.mp4` video file inside the `data/video` directory.
    * Audio: Place your `<AUDIO_FILE_NAME>.m4a` or `<AUDIO_FILE_NAME>.mp3` audio file inside the `data/audio` directory.

2. Prepare an Initial Prompt
    * Provide an initial prompt for Whisper to improve transcription accuracy (highly recommended!).

3. Run the Command
    * Execute the following command, adjusting `--threshold` as needed (optimal value may vary by video):

    * Video file
    ```shell
    uv run split split-video-file ./data/video/VIDEO_FILE_NAME.mp4 --whipser-model=turbo --initial-prompt-path=WHERE_PROMPT_SAVED.txt --threshold=2.0
    ```

    * Audieo file
    ```shell
    uv run split transcribe-audio-file ./data/audio/AUDIO_FILE_NAME.m4a --whipser-model=turbo --initial-prompt-path=WHERE_PROMPT_SAVED.txt
    ```

4. Check the Output
    * Processed transcripts will be saved in the `output/final_*` directory.


## Project Organization

```
auto_lecture_split/
├── LICENSE                     # Open-source license if one is chosen
├── README.md                   # The top-level README for developers using this project.
├── mkdocs.yml                  # mkdocs-material configuration file.
├── pyproject.toml              # Project configuration file with package metadata for
│                                   auto_lecture_split and configuration for tools like ruff
├── uv.lock                     # The lock file for reproducing the production environment, e.g.
│                                   generated with `uv sync`
├── configs                     # Config files (models and training hyperparameters)
│   └── model1.yaml
│
├── data
│
├── docs                        # Project documentation.
│
├── models                      # Trained and serialized models.
│
├── pyproject.toml              # The pyproject.toml file for reproducing the analysis environment.
├── src/tests                   # Unit test files.
│
└── src/auto_lecture_split      # Source code for use in this project.
```

## For Developers

### Whether to use `package`

This determines if the project should be treated as a Python package or a "virtual" project.

A `package` is a fully installable Python module,
while a virtual project is not installable but manages its dependencies in the virtual environment.

If you don't want to use this packaging feature,
you can set `tool.uv.package = false` in the pyproject.toml file.
This tells `uv` to handle your project as a virtual project instead of a package.

### Install Python (3.12)
```shell
uv python install 3.12
```

### Pin Python version
```shell
uv python pin 3.12
```

### Install packages with PyTorch + CUDA 12.4 (Ubuntu)
```shell
uv sync --extra cu124
```

### Install packages without locking environments
```shell
uv sync --frozen
```

### Install dev packages, too
```shell
uv sync --group dev --group docs --extra cu124
```

### Run tests
```shell
uv run pytest
```

### Linting
```shell
uvx ruff check --fix .
```

### Formatting
```shell
uvx ruff format
```

### Run pre-commit
```shell
uvx pre-commit run --all-files
```

### Build package
```shell
uv build
```

### Serve Document
```shell
uv run mkdocs serve
```

### Build Document
```shell
uv run mkdocs build
```

### Build Docker Image (from source)

[ref. uv docs](https://docs.astral.sh/uv/guides/integration/docker/#installing-a-project)

```shell
docker build -t TAGNAME -f Dockerfile.source
```

### Build Docker Image (from package)

[ref. uv docs](https://docs.astral.sh/uv/guides/integration/docker/#non-editable-installs)

```shell
docker build -t TAGNAME -f Dockerfile.package
```

### Run Docker Container
```shell
docker run --gpus all -p 8000:8000 my-production-app
```

## References
* [Packaging Python Projects](https://packaging.python.org/tutorials/packaging-projects/)
* [Python Packaging User Guide](https://packaging.python.org/)
