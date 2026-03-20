import argparse
import tomllib
import torch
import random
import numpy as np

from base.config import ExpConfig
from exp.exp_main import Exp_Main
from utils.tools import check_and_prepare_dirs, save_results, inverse_transform
#from utils.plotting import plot_results



parser = argparse.ArgumentParser(description="LTSF Runner")
parser.add_argument("--config",type=str,required=True, help="Path to experiment TOML config")
args = parser.parse_args()

# -------------------------
# Load Running Configurations.
# -------------------------
with open(args.config, "rb") as f:
    values = tomllib.load(f)


# -------------------------
# Run Experiments.
# Seed Values = 2021, 2022, 2023, 2024, 2025
# 
# -------------------------
for d_seed in range(2021, 2025):
    for run in values["runs"]:

        # -------------------------
        # Merged will allow to overide "default" settings, by settings in "runs".
        # Ex: if default.d_batch_size=512, run.d_batch_size=32, then,
        # merged.d_batch_size=32.
        # -------------------------
        merged = {
             **values['default'], 
            **values.get('gpu',{}),
            **run,
        }
        cfgs = ExpConfig(**merged)

        random.seed(d_seed)
        torch.manual_seed(d_seed)
        np.random.seed(d_seed)

        # --------------------------------
        # Set the GPU settings.
        # --------------------------------
        cfgs.d_use_gpu = True if torch.cuda.is_available() and cfgs.d_use_gpu else False

        # If using multiple GPUs, set up the device ids and the primary gpu.
        if cfgs.d_use_gpu and cfgs.d_use_multi_gpu:
            cfgs.d_devices = cfgs.devices.replace(' ', '')
            device_ids = cfgs.d_devices.split(',')
            cfgs.d_device_ids = [int(id_) for id_ in device_ids]
            cfgs.d_gpu = cfgs.device_ids[0]

        # --------------------------------
        # Print the arguments for the experiment. 
        # This is useful for logging and reproducibility.
        # --------------------------------
        print('Configuration in experiment:')
        print(cfgs)

        setting = '{}-{}-ft{}-sl{}-pl{}-seed{}'.format(
            cfgs.d_model,
            cfgs.d_data,
            cfgs.d_forecast_type,
            cfgs.d_seq_len,
            cfgs.d_pred_len,
            d_seed)

        cfgs.d_setting = setting

        # --------------------------------
        # Prepare directories.
        # --------------------------------
        check_and_prepare_dirs(cfgs)

        # --------------------------------
        # RUN EXPERIMENT.
        # --------------------------------
        exp = Exp_Main(cfgs)

        # If training is enabled.
        if cfgs.d_is_training:
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(cfgs.d_setting))
                exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(cfgs.d_setting))
        preds, trues = exp.test(cfgs.d_setting)

        # If output needs to be inverse transformed.
        if cfgs.d_inverse_transform:
            preds = inverse_transform(args, preds)
            trues = inverse_transform(args, trues)

        # --------------------------------
        # Save and plot results.
        # --------------------------------
        save_results(cfgs, preds, trues)
        #plot_results(cfgs, preds, trues, zoom_to=400)

        torch.cuda.empty_cache()