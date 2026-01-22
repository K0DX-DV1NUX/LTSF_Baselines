#!/bin/sh

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=iTransformer

root_path_name=../dataset/
train_epochs=50
patience=10
enc_in=1
features=S

data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1

for seed in 2021 3021 4021 5021 6021
do
for pred_len in 96 192 336 720
do
for seq_len in 336 512 720
do    
    if [ $pred_len -eq 96 ] || [ $pred_len -eq 192 ]; then
      d_model=256
      d_ff=256
    else
      d_model=512
      d_ff=512
    fi
    python -u run_longExp.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features $features \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --train_type nonlinear \
      --e_layers 2 \
      --enc_in $enc_in \
      --dec_in $enc_in \
      --c_out 7 \
      --d_model $d_model \
      --d_ff $d_ff \
      --train_epochs $train_epochs \
      --patience $patience \
      --des 'Exp' \
      --itr 1 \
      --batch_size 128 \
      --learning_rate 0.0001 \
      --seed $seed
done
done
done


data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2

for seed in 2021 3021 4021 5021 6021
do
for pred_len in 96 192 336 720
do
for seq_len in 336 512 720
do    
    if [ $pred_len -eq 96 ] || [ $pred_len -eq 192 ]; then
      d_model=256
      d_ff=256
    else
      d_model=512
      d_ff=512
    fi
    python -u run_longExp.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features $features \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --train_type nonlinear \
      --e_layers 2 \
      --enc_in $enc_in \
      --dec_in $enc_in \
      --c_out 7 \
      --d_model $d_model \
      --d_ff $d_ff \
      --train_epochs $train_epochs \
      --patience $patience \
      --des 'Exp' \
      --itr 1 \
      --batch_size 128 \
      --learning_rate 0.0001 \
      --seed $seed
done
done
done

data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1

for seed in 2021 3021 4021 5021 6021
do
for pred_len in 96 192 336 720
do
for seq_len in 336 512 720
do    
    if [ $pred_len -eq 96 ] || [ $pred_len -eq 192 ]; then
      d_model=256
      d_ff=256
    else
      d_model=512
      d_ff=512
    fi
    python -u run_longExp.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features $features \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --train_type nonlinear \
      --e_layers 2 \
      --enc_in $enc_in \
      --dec_in $enc_in \
      --c_out 7 \
      --d_model $d_model \
      --d_ff $d_ff \
      --train_epochs $train_epochs \
      --patience $patience \
      --des 'Exp' \
      --itr 1 \
      --batch_size 128 \
      --learning_rate 0.0001 \
      --seed $seed
done
done
done

data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2

for seed in 2021 3021 4021 5021 6021
do
for pred_len in 96 192 336 720
do
for seq_len in 336 512 720
do    
    if [ $pred_len -eq 96 ] || [ $pred_len -eq 192 ]; then
      d_model=256
      d_ff=256
    else
      d_model=512
      d_ff=512
    fi
    python -u run_longExp.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features $features \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --train_type nonlinear \
      --e_layers 2 \
      --enc_in $enc_in \
      --dec_in $enc_in \
      --c_out 7 \
      --d_model $d_model \
      --d_ff $d_ff \
      --train_epochs $train_epochs \
      --patience $patience \
      --des 'Exp' \
      --itr 1 \
      --batch_size 128 \
      --learning_rate 0.0001 \
      --seed $seed
done
done
done
