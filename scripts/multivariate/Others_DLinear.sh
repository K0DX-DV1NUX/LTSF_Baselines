#!/bin/sh

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

# Default settings being used across all scripts.
root_path_name=../dataset/
train_epochs=100
patience=20
learning_rate=0.001
batch_size=512
stride=1
lradj=type7
inverse_transform=0
num_workers=0

# Model specific settings.
model_name=DLinear
model_input_type="x_only"
individual=1

run_experiment () {
    data_path_name=$1
    model_id_name=$2
    data_name=$3
    forecast_type=$4
    in_features=$5

    for seed in $(seq 2021 2025)
    do
        for pred_len in 48 96 192 336 512 720
        do
            for seq_len in 336
            do    
                python -u run_longExp.py \
                --d_is_training 1 \
                --d_seed $seed \
                --d_root_path $root_path_name \
                --d_data_path $data_path_name \
                --d_data $data_name \
                --d_forecast_type $forecast_type \
                --d_model_input_type $model_input_type \
                --d_seq_len $seq_len \
                --d_pred_len $pred_len \
                --d_in_features $in_features \
                --d_batch_size $batch_size \
                --d_stride $stride \
                --d_train_epochs $train_epochs \
                --d_patience $patience \
                --d_learning_rate $learning_rate \
                --d_lradj $lradj \
                --d_num_workers $num_workers \
                --d_model $model_name \
                --m_individual $individual
            done
        done
    done
}

run_experiment weather.csv Weather WTH M 21
run_experiment electricity.csv Electricity ELT M 321
run_experiment traffic.csv Traffic TRF M 862