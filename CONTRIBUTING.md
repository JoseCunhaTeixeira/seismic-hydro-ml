# Contributing

Thanks for your interest in improving this project.

## Development setup

```bash
git clone https://github.com/JoseCunhaTeixeira/seismic-hydro-ml.git
cd seismic-hydro-ml
python -m venv .venv
source .venv/bin/activate  # .venv\Scripts\activate on Windows
pip install -e ".[dev]"
pre-commit install  # optional, runs lint/format checks on commit
```

## Before opening a pull request

```bash
ruff check .          # lint
ruff format .         # format
pytest                # unit tests
```

CI runs the same checks on every pull request and must pass before merging.

## Guidelines

- Keep pull requests focused on a single change.
- Add or update tests for any change in `src/seismic_hydro_ml/`.
- Do not commit large binary files (raw data, trained models) outside of what
  is already tracked under `input/`, `models/`, and `output/` — see the
  [README](README.md#data--models) for how those directories are used.
- Scientific/methodological changes (e.g. to the network architecture or the
  feature engineering in `data_prep.py`) should reference the
  [associated paper](https://doi.org/10.1029/2024WR037706) or explain the
  rationale in the pull request description.

## Reporting issues

Please open a [GitHub issue](https://github.com/JoseCunhaTeixeira/seismic-hydro-ml/issues)
with a clear description, steps to reproduce, and, if applicable, the
error traceback.
