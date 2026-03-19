#!/bin/sh

# Default settings being used across all scripts.
root_path_name=../dataset/
train_epochs=100
patience=20
learning_rate=0.001
stride=5
lradj=type7
inverse_transform=0
num_workers=10

# Model specific settings.
model_name=PatchTST
model_input_type="x_only"

run_experiment () {
    data_path_name=$1
    model_id_name=$2
    data_name=$3
    forecast_type=$4
    in_features=$5
    batch_size=$6

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
                --m_individual 1 \
                --m_e_layers 3 \
                --m_n_heads 4 \
                --m_d_model 16 \
                --m_d_ff 128 \
                --m_dropout 0.3 \
                --m_fc_dropout 0.3 \
                --m_head_dropout 0 \
                --m_patch_len 16 \
                --m_stride 8\
                --m_revin 1 \
                --m_affine 0 \
                --m_subtract_last 0 \
                --m_decomposition 0 \
                --m_kernel_size 25 \
                --m_padding_patch end
            done
        done
    done
}

#run_experiment ETTh1.csv ETTh1 ETTh1 M 7 512
#run_experiment ETTh2.csv ETTh2 ETTh2 M 7 512
#run_experiment ETTm1.csv ETTm1 ETTm1 M 7 512
#run_experiment ETTm2.csv ETTm2 ETTm2 M 7 512
#run_experiment weather.csv Weather WTH M 21 512
run_experiment electricity.csv Electricity ELT M 321 32
run_experiment traffic.csv Traffic TRF M 862 32