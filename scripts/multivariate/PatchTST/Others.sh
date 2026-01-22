#!/bin/sh

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi


root_path_name=../dataset/

model_name=PatchTST
train_epochs=100
patience=20
num_workers=0
learning_rate=0.01
batch_size=128
lradj=type7



run_experiment () {
    data_path_name=$1
    model_id_name=$2
    data_name=$3
    features=$4
    enc_in=$5

    for seed in $(seq 2021 2025)
    do
        for pred_len in 48 96 192 336 512 720
        do
            for seq_len in 192 336 512
            do    
                python -u run_longExp.py \
                  --is_training 1 \
                  --individual 1 \
                  --seed $seed \
                  --root_path $root_path_name \
                  --data_path $data_path_name \
                  --model_id $model_id_name \
                  --model $model_name \
                  --data $data_name \
                  --features $features \
                  --train_type Linear \
                  --seq_len $seq_len \
                  --pred_len $pred_len \
                  --enc_in $enc_in \
                  --e_layers 3 \
                  --n_heads 4 \
                  --d_model 16 \
                  --d_ff 128 \
                  --dropout 0.3\
                  --fc_dropout 0.3\
                  --head_dropout 0\
                  --patch_len 16\
                  --stride 8\
                  --train_epochs $train_epochs \
                  --patience $patience \
                  --des 'Exp' \
                  --itr 1 \
                  --batch_size $batch_size \
                  --learning_rate $learning_rate \
                  --lradj $lradj \
                  --num_workers $num_workers
            done
        done
    done
}

run_experiment electricity.csv Electricity custom M 321
run_experiment traffic.csv Traffic custom M 862
run_experiment weather.csv Weather custom M 21