import argparse
import csv
import random
import tomllib
from pathlib import Path

import numpy as np
import torch
from fvcore.nn import FlopCountAnalysis

from base.config import ExpConfig
from models import DLinear, FrNet, HaDAM, MeanMedian, ModernTCN, Naive, PatchTST, SparseTSF, iTransformer


MODEL_DICT = {
    "DLinear": DLinear,
    "MeanMedian": MeanMedian,
    "Naive": Naive,
    "PatchTST": PatchTST,
    "SparseTSF": SparseTSF,
    "iTransformer": iTransformer,
    "ModernTCN": ModernTCN,
    "FrNet": FrNet,
    "ModelX": HaDAM,
}


parser = argparse.ArgumentParser(description="LTSF Model Profiler")
parser.add_argument("--data_config", type=str, required=True, help="Path to dataset TOML config.")
parser.add_argument("--gpu_config", type=str, required=True, help="Path to gpu TOML config.")
parser.add_argument("--model_config", type=str, required=True, help="Path to model TOML config.")
args = parser.parse_args()


def load_toml(path):
    with open(path, "rb") as f:
        return tomllib.load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_time_feature_dim(cfg):
    freq = cfg.d_freq.lower()
    if cfg.d_embed == "timeF":
        freq_map = {
            "h": 4,
            "t": 5,
            "min": 5,
            "minute": 5,
            "s": 6,
            "m": 1,
            "a": 1,
            "w": 2,
            "d": 3,
            "b": 3,
        }
        if freq not in freq_map:
            raise ValueError(f"Unsupported d_freq '{cfg.d_freq}' for timeF embedding.")
        return freq_map[freq]

    dims = 4
    if freq in {"t", "min", "minute", "s"}:
        dims += 1
    if freq == "s":
        dims += 1
    return dims


def build_dummy_mark(batch_size, seq_len, cfg):
    mark_dim = get_time_feature_dim(cfg)
    if cfg.d_embed == "timeF":
        return torch.randn(batch_size, seq_len, mark_dim)
    return torch.zeros(batch_size, seq_len, mark_dim, dtype=torch.long)


def build_inputs(cfg, batch_size):
    x = torch.randn(batch_size, cfg.d_seq_len, cfg.d_in_features)
    input_type = cfg.d_model_input_type.lower()

    if input_type == "x_only":
        return (x,)

    x_mark = build_dummy_mark(batch_size, cfg.d_seq_len, cfg)

    if input_type == "x_mark_incl":
        return (x, x_mark)

    if input_type == "x_mark_dec_incld":
        y_seq_len = cfg.d_label_len + cfg.d_pred_len
        dec_feature_dim = cfg.d_in_features
        dec_inp = torch.randn(batch_size, y_seq_len, dec_feature_dim)
        y_mark = build_dummy_mark(batch_size, y_seq_len, cfg)
        return (x, x_mark, dec_inp, y_mark)

    raise ValueError(
        f"Unsupported d_model_input_type '{cfg.d_model_input_type}'. "
        "Supported values are x_only, x_mark_incl, and x_mark_dec_incld."
    )


def build_model(cfg):
    if cfg.d_model not in MODEL_DICT:
        raise ValueError(f"Unsupported model '{cfg.d_model}'. Add it to MODEL_DICT in runParams.py.")
    if cfg.d_model == "SparseTSF":
        cfg.seq_len = cfg.d_seq_len
        cfg.pred_len = cfg.d_pred_len
        cfg.enc_in = cfg.d_in_features
        cfg.d_model = getattr(cfg, "m_d_model", 128)
        cfg.period_len = cfg.m_period_len
        cfg.model_type = cfg.m_model_type
    return MODEL_DICT[cfg.d_model].Model(cfg).float().eval()


def get_model_size_mb(model):
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    total_bytes = param_bytes + buffer_bytes
    return total_bytes / (1024 ** 2), total_bytes


def resolve_profile_output(profile_cfg, model_config_path):
    model_dir = model_config_path.resolve().parent.parent.name
    output_dir = Path(profile_cfg.get("output_dir", "./Results_Params"))
    output_name = profile_cfg.get("output_name", f"Result_{model_dir}.csv")
    return output_dir / output_name


def upsert_rows(csv_path, new_rows):
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    existing = {}
    if csv_path.exists():
        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (
                    row["profile_name"],
                    row["dataset"],
                    row["seq_len"],
                    row["pred_len"],
                    row["model_config"],
                )
                existing[key] = row

    for row in new_rows:
        key = (
            row["profile_name"],
            row["dataset"],
            str(row["seq_len"]),
            str(row["pred_len"]),
            row["model_config"],
        )
        existing[key] = row

    fieldnames = [
        "profile_name",
        "model",
        "dataset",
        "forecast_type",
        "seq_len",
        "pred_len",
        "in_features",
        "out_features",
        "total_params",
        "trainable_params",
        "non_trainable_params",
        "flops",
        "macs_estimated",
        "model_size_mb",
        "model_size_bytes",
        "batch_size",
        "seed",
        "status",
        "notes",
        "data_config",
        "model_config",
    ]

    sorted_rows = sorted(
        existing.values(),
        key=lambda row: (
            row["dataset"],
            int(row["seq_len"]),
            int(row["pred_len"]),
            row["model_config"],
        ),
    )

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted_rows)


data_values = load_toml(args.data_config)
gpu_values = load_toml(args.gpu_config)
model_values = load_toml(args.model_config)

profile_cfg = model_values.get("profile", {})
profile_seed = profile_cfg.get("seed", 2021)
profile_batch_size = profile_cfg.get("batch_size", 1)
profile_name = profile_cfg.get("profile_name", Path(args.model_config).resolve().stem)
csv_path = resolve_profile_output(profile_cfg, Path(args.model_config))

rows = []

for run in model_values["runs"]:
    merged = {
        **data_values["default"],
        **gpu_values.get("gpu", {}),
        **model_values["default"],
        **run,
    }

    cfgs = ExpConfig(**merged)
    cfgs.d_use_gpu = False

    set_seed(profile_seed)

    row = {
        "profile_name": profile_name,
        "model": cfgs.d_model,
        "dataset": cfgs.d_data,
        "forecast_type": cfgs.d_forecast_type,
        "seq_len": cfgs.d_seq_len,
        "pred_len": cfgs.d_pred_len,
        "in_features": cfgs.d_in_features,
        "out_features": cfgs.d_out_features,
        "batch_size": profile_batch_size,
        "seed": profile_seed,
        "status": "ok",
        "notes": "MACs estimated as FLOPs / 2 from fvcore output.",
        "data_config": args.data_config,
        "model_config": args.model_config,
    }

    try:
        model = build_model(cfgs)
        inputs = build_inputs(cfgs, profile_batch_size)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_size_mb, model_size_bytes = get_model_size_mb(model)

        flop_analysis = FlopCountAnalysis(model, inputs)
        flop_analysis.unsupported_ops_warnings(False)
        flop_analysis.uncalled_modules_warnings(False)
        flops = int(flop_analysis.total())
        macs_estimated = flops / 2.0

        row.update(
            {
                "total_params": total_params,
                "trainable_params": trainable_params,
                "non_trainable_params": total_params - trainable_params,
                "flops": flops,
                "macs_estimated": macs_estimated,
                "model_size_mb": round(model_size_mb, 6),
                "model_size_bytes": model_size_bytes,
            }
        )
    except Exception as exc:
        row.update(
            {
                "total_params": "",
                "trainable_params": "",
                "non_trainable_params": "",
                "flops": "",
                "macs_estimated": "",
                "model_size_mb": "",
                "model_size_bytes": "",
                "status": "failed",
                "notes": str(exc),
            }
        )

    rows.append(row)
    print(
        f"[{row['status']}] {cfgs.d_model} | {cfgs.d_data} | "
        f"seq={cfgs.d_seq_len} pred={cfgs.d_pred_len} | "
        f"params={row['total_params']} flops={row['flops']} size_mb={row['model_size_mb']}"
    )

upsert_rows(csv_path, rows)
print(f"Saved profiling results to: {csv_path}")
