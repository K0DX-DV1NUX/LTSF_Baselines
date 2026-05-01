# BASELINES

This repository contains a configurable long-term time-series forecasting
benchmark runner. It trains and evaluates baseline forecasting models across
standard datasets such as ETT, Weather, Electricity, and Traffic using TOML
configuration files.

The code is organized around one experiment runner, `runExp.py`, which combines
dataset defaults, GPU/runtime settings, and model-specific hyperparameters into
a single `ExpConfig` object before launching training and testing.

## Supported Models

The experiment runner currently registers these models in `exp/exp_main.py`:

- `DLinear`
- `FrNet`
- `HaDAM`
- `MeanMedian`
- `ModernTCN`
- `Naive`
- `PatchTST`
- `SparseTSF`
- `iTransformer`

Model implementations live in `models/`, with shared layers in `layers/`.

## Repository Layout

```text
base/                  Pydantic experiment configuration schema
data_provider/          Time-series dataset loader and train/val/test splitting
exp/                    Training, validation, and testing loop
layers/                 Shared neural network layers
models/                 Forecasting model implementations
scripts/defaults/       Dataset and GPU TOML defaults
scripts/<model>/        Model-specific run scripts and TOML configs
utils/                  Metrics, learning-rate logic, plotting, and IO helpers
runExp.py               Main train/test entry point
runParams.py            Parameter, FLOP, MAC, and model-size profiler
```

## Setup

Create an environment and install the Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The project expects datasets under `../dataset` by default. You can change this
per dataset in the corresponding TOML file under `scripts/defaults/`.

Each CSV is expected to contain:

- a `date` column
- one or more numeric time-series feature columns
- the target column, selected by `d_target` for univariate or
  multivariate-to-univariate forecasting

## Running Experiments

Run one model on one dataset:

```bash
python -u runExp.py \
  --gpu_config scripts/defaults/gpu.toml \
  --data_config scripts/defaults/ETTh1.toml \
  --model_config scripts/DLinear/toml/model.toml
```

Run all default datasets for a model using its shell script:

```bash
sh scripts/DLinear/run.sh
```

`runExp.py` repeats each run for seeds `2021` through `2025`. For every seed and
run block, it:

1. merges the TOML configuration files
2. builds an `ExpConfig`
3. prepares train, validation, and test loaders
4. trains the selected model if `d_is_training = 1`
5. evaluates on the test split
6. saves predictions and ground truth arrays

## Configuration

Experiments are configured by three TOML files:

- `--data_config`: dataset, feature dimensions, frequency, training defaults
- `--gpu_config`: worker and GPU settings
- `--model_config`: model identity, model hyperparameters, and run grid

The merge priority is:

```text
data_config[default] < gpu_config[gpu] < model_config[default] < [[runs]]
```

Values on the right override values on the left. This means a dataset file can
define broad defaults, a model file can override model-relevant defaults, and
each `[[runs]]` block can override only the values that change for that run.

Example:

```toml
[default]
d_model = "HaDAM"
d_model_input_type = "x_only"
m_patch_size = 8
m_patch_dim = 128

[[runs]]
d_seq_len = 336
d_pred_len = 96
d_test_stride = 96
```

### The `d_` and `m_` Naming Style

Configuration keys use prefixes to make their ownership clear.

`d_` means "data/driver/experiment" configuration. These values are consumed by
the shared experiment system: dataset loading, sequence lengths, forecast type,
training behavior, optimizer settings, GPU settings, checkpoints, and output
behavior.

Common `d_` keys include:

- `d_model`: model name used by the runner
- `d_model_input_type`: model call signature used by the training loop
- `d_data`, `d_root_path`, `d_data_path`: dataset identity and location
- `d_forecast_type`: `M`, `S`, or `MS`
- `d_seq_len`, `d_label_len`, `d_pred_len`: input, decoder-label, and forecast lengths
- `d_in_features`, `d_out_features`: channel dimensions
- `d_train_epochs`, `d_batch_size`, `d_learning_rate`, `d_patience`: training settings
- `d_use_gpu`, `d_gpu`, `d_use_multi_gpu`, `d_devices`, `d_use_amp`: device settings

`m_` means "model-local" configuration. These values are read inside individual
model implementations and usually describe architecture choices or
model-specific behavior. The shared runner passes them through because
`ExpConfig` allows extra fields.

Common `m_` examples include:

- `m_patch_len`, `m_patch_size`, `m_stride`: patching settings
- `m_d_model`, `m_d_ff`, `m_n_heads`, `m_e_layers`: Transformer-style dimensions
- `m_dropout`, `m_fc_dropout`, `m_head_dropout`: dropout settings
- `m_revin`, `m_affine`, `m_subtract_last`: normalization settings
- `m_period_list`, `m_global_freq_pred`: frequency/period settings used by FrNet
- `m_stat_type`, `m_naive_type`: simple baseline behavior

This split keeps shared experiment controls separate from architecture-specific
hyperparameters. When adding a new model, prefer `d_` only for values the runner,
dataset, or training loop must understand; use `m_` for values that only the new
model reads.

## Forecasting Modes

`d_forecast_type` controls which variables are used and predicted:

- `M`: multivariate input to multivariate output
- `S`: univariate input to univariate output
- `MS`: multivariate input to univariate output

For `S` and `MS`, `d_target` selects the target column. It can be an integer
index into the CSV columns or a column name.

## Model Input Types

`d_model_input_type` tells the training loop how to call the model:

- `x_only`: pass only the encoder input `x`
- `x_mark_incl`: pass `x` and encoder time features
- `x_mark_dec_incld`: pass `x`, encoder time features, decoder input, and decoder
  time features

Use the value expected by the selected model. Most simple baselines use
`x_only`; `iTransformer` uses `x_mark_dec_incld` in the provided configs.

## Outputs

For each experiment setting, the runner writes:

- `checkpoints/<setting>/checkpoint.pth`: best model checkpoint
- `checkpoints/<setting>/scaler.pkl`: fitted scaler used for normalized data
- `results/<setting>/preds.npy`: flattened test predictions
- `results/<setting>/trues.npy`: flattened test ground truth
- `plots/<setting>/`: plots when `d_plot_results = true`
- `Result_<model>.csv`: appended summary lines with MSE and MAE

The setting name is built as:

```text
<model>-<dataset>-ft<forecast_type>-sl<seq_len>-pl<pred_len>-seed<seed>
```

## Profiling Parameters and FLOPs

`runParams.py` profiles model size, parameter counts, FLOPs, and estimated MACs
for every `[[runs]]` block in a model config:

```bash
python -u runParams.py \
  --gpu_config scripts/defaults/gpu.toml \
  --data_config scripts/defaults/ETTh1.toml \
  --model_config scripts/DLinear/toml/model.toml
```

Profiling results are written to `Results_Params/` by default, unless a model
TOML provides a `[profile]` section with a custom `output_dir` or `output_name`.

## Adding a New Model

1. Add the model implementation under `models/`.
2. Expose a `Model(configs)` class.
3. Add the model to `model_dict` in `exp/exp_main.py`.
4. Add it to `MODEL_DICT` in `runParams.py` if profiling is needed.
5. Create a model TOML under `scripts/<ModelName>/toml/`.
6. Use `d_` keys for shared runner settings and `m_` keys for model-local
   hyperparameters.

## Notes

- The dataset split is fixed at `70%` train, `10%` validation, and `20%` test in
  `TimeSeriesDataset`.
- Training windows use `d_train_stride`; validation and test windows use
  `d_test_stride`.
- Data is standardized with `StandardScaler` fitted on the training split and
  saved alongside the checkpoint.
- Test output is flattened from forecast windows into a continuous sequence
  before metrics are computed.
