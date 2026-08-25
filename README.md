# seismic-hydro-ml

[![CI](https://github.com/JoseCunhaTeixeira/seismic-hydro-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/JoseCunhaTeixeira/seismic-hydro-ml/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Paper DOI](https://img.shields.io/badge/DOI-10.1029%2F2024WR037706-blue.svg)](https://doi.org/10.1029/2024WR037706)

Predicting groundwater table (GWT) level maps from passive seismic surface-wave
dispersion curves, using a Multilayer Perceptron (MLP) trained on real field
data. Daily 2D GWT maps are extrapolated across a seismic array from a single
piezometric reference point.

![Groundwater table prediction](https://github.com/user-attachments/assets/6395b8d9-c7f4-4572-9cff-431bbd5d474e)
![Example result](https://github.com/JoseCunhaTeixeira/MLP_GWT_prediction/assets/148117375/9680214f-6188-4328-a106-f9a48f338828)

This code accompanies the paper:

> Cunha Teixeira, J., Bodet, L., Rivière, A., Hallier, A., Gesret, A., Dangeard, M.,
> Dhemaied, A., & Boisson Gaboriau, J. (2025). *Physics-Guided Deep Learning Model
> for Daily Groundwater Table Maps Estimation Using Passive Surface-Wave Dispersion*.
> Water Resources Research, 61(1), e2024WR037706.
> [https://doi.org/10.1029/2024WR037706](https://doi.org/10.1029/2024WR037706)

See [CITATION.cff](CITATION.cff) for citation metadata.

## Overview

The pipeline has four stages, each exposed as a console command:

| Stage | Command | Description |
|---|---|---|
| 1. Data preparation | `shml-prepare-data` | Build wavelength-domain dispersion-curve arrays and aligned piezometer targets from raw survey data. Requires internal SNCF Réseau raw data — see [Data & models](#data--models). |
| 2. Training | `shml-train` | Train the MLP on the prepared arrays and save the fitted model. |
| 3. Evaluation | `shml-evaluate` | Evaluate a trained model against the piezometer time series at survey points and plot predicted vs. observed GWT level. |
| 4. Map building | `shml-build-maps` | Run the trained model over the full survey grid for every day to produce 2D daily GWT maps, then sanity-check them against the piezometers. |

Stages 2-4 run out of the box on the pre-built arrays already committed under
[`input/`](input). Stage 1 is included for data-lineage transparency but needs
raw data that is not distributed with this repository.

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/JoseCunhaTeixeira/seismic-hydro-ml.git
cd seismic-hydro-ml
python -m venv .venv
source .venv/bin/activate  # .venv\Scripts\activate on Windows
pip install -e .
```

For development (tests, linting, pre-commit hooks), install the extras:

```bash
pip install -e ".[dev]"
```

## Usage

Train a model on the committed dataset:

```bash
shml-train --epochs 500 --batch-size 2
```

This saves `<run_name>_MLP.sav` and `<run_name>_history.png` under `models/`.

Evaluate a trained model against the piezometer time series:

```bash
shml-evaluate --model-name 20240124-1903_MLP_trainPZ3
```

Build daily GWT maps over the full survey grid and check them against the
piezometers:

```bash
shml-build-maps --model-name 20240124-1903_MLP_trainPZ3
```

Pass `--skip-predict` to `shml-build-maps` to re-run the piezometer check
against an already-computed map without recomputing predictions.

Run any command with `--help` to see the full list of options (learning rate,
epochs, hidden layer size, early-stopping patience, input/output directories,
etc.).

### Data preparation

`shml-prepare-data` rebuilds everything under `input/` from raw data. It
requires:

- A directory of raw Rayleigh-wave velocity cube files, read with the
  internal `ndimcube` package (not published with this repository).
- A directory of `PZ3_interp300s.csv` / `PZ5_interp300s.csv` piezometer time
  series.

```bash
export SHML_RAW_VR_DIR=/path/to/raw/vr/cubes
export SHML_RAW_PIEZO_DIR=/path/to/piezometer/csvs
export SHML_NDIMCUBE_PATH=/path/to/ndimcube/parent/dir
shml-prepare-data
```

(equivalently, pass `--raw-vr-dir`, `--raw-piezo-dir`, `--ndimcube-path`).

## Data & models

| Directory | Contents |
|---|---|
| `input/` | Prepared training/validation arrays and dispersion-curve database, already committed so stages 2-4 run without access to raw data. |
| `models/` | Trained models (`*.sav`) and their training-loss curves (`*.png`). |
| `output/` | Predicted GWT maps (`*.npy`) and evaluation plots (`*.png`). |

All three default to the corresponding directory at the repository root and
can be overridden per-command (`--input-dir`, `--models-dir`, `--output-dir`)
or globally via the `SHML_INPUT_DIR`, `SHML_MODELS_DIR`, `SHML_OUTPUT_DIR`
environment variables — see [`config.py`](src/seismic_hydro_ml/config.py).

## Project structure

```
seismic-hydro-ml/
├── src/seismic_hydro_ml/
│   ├── config.py       # data-directory configuration (env-var overridable)
│   ├── data_prep.py     # stage 1: raw data -> input/ arrays
│   ├── model.py          # the MLP
│   ├── train.py           # stage 2: train + save a model
│   ├── evaluate.py         # stage 3: evaluate a model at survey points
│   └── build_maps.py        # stage 4: full-grid daily GWT maps
├── tests/                     # unit tests (pytest)
├── input/ models/ output/      # data (see Data & models)
└── .github/workflows/ci.yml     # lint + test on every push/PR
```

## Testing

```bash
pytest
```

CI (`.github/workflows/ci.yml`) runs `ruff check` and the test suite on
Python 3.10-3.12 for every push and pull request to `main`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE) © José Cunha Teixeira.

This project was developed as part of a PhD at SNCF Réseau, Sorbonne
Université, and Mines Paris - PSL.
