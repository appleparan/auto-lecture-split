# auto-lecture-split




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
│   ├── external                # Data from third party sources.
│   ├── interim                 # Intermediate data that has been transformed.
│   ├── processed               # The final, canonical data sets for modeling.
│   └── raw                     # The original, immutable data dump.
│
├── docs                        # Project documentation.
│
├── models                      # Trained and serialized models.
│
├── notebooks                   # Jupyter notebooks.
│
├── references                  # Data dictionaries, manuals, and all other explanatory materials.
│
├── reports                     # Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures                 # Generated graphics and figures to be used in reporting.
│
├── pyproject.toml              # The pyproject.toml file for reproducing the analysis environment.
├── src/tests                   # Unit test files.
│
└── src/auto_lecture_split      # Source code for use in this project.
    │
    ├── __init__.py             # Makes auto_lecture_split a Python module
    │
    ├── cli.py                  # Default CLI program
    │
    ├── data                    # Data engineering scripts.
    │
    ├── models                  # ML model engineering (a folder for each model).
    │
    └── visualization           # Scripts to create exploratory and results oriented visualizations.
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
uvx ruff fmt
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
