# Forest Cover Classification with an MLP

Predict one of seven forest cover types from cartographic measurements. This
project implements Deep Learning Assignment 1 using the [UCI Covertype
dataset](https://archive.ics.uci.edu/dataset/31/covertype): 581,012 records,
54 raw features, and seven classes. The notebook explains the problem, EDA,
preprocessing, neural network, 36 experiments, evaluation, and limitations.

## Set up on Windows

The project was verified with Python 3.14.6 on a CPU. A GPU is not required.
From PowerShell in the repository folder, create an environment and install
the dependencies:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m notebook covertype_mlp.ipynb
```

Select the environment's Python kernel, then use **Restart Kernel and Run All
Cells**. The first data-loading cell downloads Covertype into `data/`; an
internet connection is needed only if the dataset is not cached. The complete
dataset is used, with a stratified 70% training / 15% validation / 15% test split
and seed 42. Training time depends on the CPU and worker count; allow roughly
40-60 minutes for a fresh run. Finished experiment caches can shorten reruns.

To execute the equivalent script:

```powershell
.\.venv\Scripts\python.exe covertype_mlp.py
```

Plots are shown during execution. For a headless script run, set
`$env:MPLBACKEND = "Agg"` first. Outputs are saved under `figures/`, `results/`,
and `models/`. Keep `forest_cover.py` beside the notebook/script. Running either replaces generated artifacts
and appends to the training log. `N_JOBS` in section 6 controls worker count;
section 7 uses two workers. Set both to 1 for lower memory use.
Unfinished experiments receive a separate data copy per worker. Cached experiments
load in the main process without launching workers; automatic joblib memory mapping
is disabled to avoid Windows resource-tracker cleanup errors.

## Predict without retraining

The repository includes the trained checkpoint and seven sample input rows.
Prediction works offline and does not load the full dataset:

```powershell
.\.venv\Scripts\python.exe predict.py examples/sample_input.csv
.\.venv\Scripts\python.exe predict.py examples/sample_input.csv --output predictions.csv
```

To use another CSV, retain the **54 named input columns** shown in the example:
10 numeric cartographic measurements, `Wilderness_Area_0` through
`Wilderness_Area_3`, and `Soil_Type_0` through `Soil_Type_39`. Each categorical
group must contain exactly one 1 and otherwise 0. Column order can vary.
An optional `Cover_Type` target and other extra columns are ignored. Missing
numeric measurements use saved training medians; missing categorical indicators,
non-numeric values, infinity, and invalid physical ranges produce an error.

The pipeline derives sine/cosine aspect and Euclidean distance to water, imputes
the resulting continuous features, clips to training quantiles, and applies the
saved scaler. It creates the same 56 inputs used during training. No statistics
are fitted on prediction data. Output rows retain input order and include:

- `cover_type`: original UCI class label, 1 through 7.
- `class_name`: corresponding species/group name.
- `confidence`: largest softmax probability, without a calibration guarantee.
- `probability_1` through `probability_7`: probabilities in UCI label order.

`--model PATH` selects another format-2 checkpoint; `--batch-size N` controls
inference batch size. The default model path is relative to `predict.py`, so
inference also works when invoked from a different folder.

## Recorded results

Model selection uses **validation macro F1** because class frequencies differ
by about 103:1. The selected run is `I1_final_60ep_no_es`: a
56 -> 256 -> 128 -> 64 -> 7 MLP with tanh, BatchNorm, Xavier initialization,
Adam, one-cycle maximum learning rate 0.003, batch size 512, and 60 epochs.
Dropout, weight decay, L1, and class weights are off.

| Metric | Baseline test | Final test |
|---|---:|---:|
| Accuracy | 0.759019 | 0.959209 |
| Macro precision | 0.735401 | 0.936267 |
| Macro recall | 0.497135 | 0.929789 |
| Macro F1 | 0.518779 | 0.932991 |
| Weighted F1 | 0.749017 | 0.959181 |
| Macro one-vs-rest ROC-AUC | 0.947858 | 0.998115 |

The final validation macro F1 is 0.931955. Both early-stopping candidates ran
all 60 epochs; patience was never exhausted. The enabled candidate restored
epoch 59, while epoch 60 without early stopping had slightly higher validation
macro F1. The final test set contains 87,152 records and was reserved from tuning.

The notebook documents which techniques helped or hurt and separates observations
from possible explanations. Several comparisons change a bundle of settings;
they do not establish the effect of each setting independently. A single seed,
greedy search, and random splits of geographically related cells limit the
conclusions. Predictions have not been validated for operational forestry use
or transfer to new regions. Repeated seeds, spatial validation with suitable
metadata, stronger tabular baselines, and calibration are future improvements.

## Verify the project

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests cover train-only imputation, input validation, feature engineering,
saved-model predictions, batching, column order, and the prediction CLI.
Recorded metrics were also independently reproduced from the saved model on
the full test split. Different package versions or hardware may produce different
results when retraining; the seed alone does not guarantee identical results.

## Files and reproducibility

| Path | Purpose |
|---|---|
| `covertype_mlp.ipynb` | Executed assignment notebook with plots, tables, and observations |
| `covertype_mlp.py` | Equivalent script in percent-cell format |
| `forest_cover.py` | Shared MLP and preprocessing helpers; safe to import |
| `predict.py` | CSV prediction CLI and Python predictor |
| `examples/sample_input.csv` | Seven raw Covertype records without target labels |
| `models/final_mlp.pt` | Weights, configuration, feature order, and fitted preprocessing |
| `results/experiments.csv` | All 36 training/validation experiment comparisons |
| `results/baseline_vs_final.csv` | Separate train, validation, and test metrics |
| `results/training_log.txt` | Per-epoch logs from the recorded training runs |
| `figures/` | EDA, experiment curves, confusion matrices, ROC, and class F1 plots |
| `tests/test_pipeline.py` | Automated pipeline and CLI checks |

The format-2 checkpoint stores tensor weights plus plain Python metadata,
including training medians, clipping limits, and scaling statistics. It can be
loaded with `torch.load(..., weights_only=True)`. The supplied trained weights
were retained when adding the missing preprocessing metadata.

`data/` and `results/cache/` are intentionally excluded from Git. Experiment
caches are reused by name/configuration. If you change data, preprocessing,
training code, or library versions, set `USE_CACHE = False` in section 6 to
retrain rather than reuse stale experiments. With unchanged inputs, section 5
still trains the baseline even when later experiments are cached.
