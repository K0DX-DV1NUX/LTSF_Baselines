#!/bin/sh

model_config="scripts/FrNet/toml/"
data_config="scripts/defaults/"
gpu_config="scripts/defaults/gpu.toml"

python -u runExp.py --gpu_config $gpu_config --data_config "$data_config/ETTh1.toml" --model_config "$model_config/ETTh1.toml"
python -u runExp.py --gpu_config $gpu_config --data_config "$data_config/ETTh2.toml" --model_config "$model_config/ETTh2.toml"
python -u runExp.py --gpu_config $gpu_config --data_config "$data_config/ETTm1.toml" --model_config "$model_config/ETTm1.toml"
python -u runExp.py --gpu_config $gpu_config --data_config "$data_config/ETTm2.toml" --model_config "$model_config/ETTm2.toml"
python -u runExp.py --gpu_config $gpu_config --data_config "$data_config/WTH.toml" --model_config "$model_config/WTH.toml"
python -u runExp.py --gpu_config $gpu_config --data_config "$data_config/ELT.toml" --model_config "$model_config/ELT_TRF.toml"
python -u runExp.py --gpu_config $gpu_config --data_config "$data_config/TRF.toml" --model_config "$model_config/ELT_TRF.toml"
