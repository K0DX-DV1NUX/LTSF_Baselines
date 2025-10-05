if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=SparseTSF
root_path_name=../dataset/
batch_size=128
train_epochs=100
patience=20
lr_rate=0.001

for period_len in 4 8 16 24 32 48
do
  data_path_name=ETTh1.csv
  model_id_name=ETTh1
  data_name=ETTh1


  for seed in 2023 2024 2025 2026 2027
  do
  for seq_len in 336 512 720
  do
  for pred_len in 48 96 192 336 512 720
  do
    python -u run_longExp.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_sl'$seq_len'_pl'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --period_len $period_len \
      --model_type 'mlp' \
      --d_model 128 \
      --enc_in 7 \
      --train_epochs $train_epochs \
      --patience $patience \
      --itr 1 \
      --batch_size $batch_size \
      --learning_rate $lr_rate \
      --lradj type7 \
      --seed $seed
  done
  done
  done


  data_path_name=ETTh2.csv
  model_id_name=ETTh2
  data_name=ETTh2

  for seed in 2023 2024 2025 2026 2027
  do
  for seq_len in 336 512 720
  do
  for pred_len in 48 96 192 336 512 720
  do
    python -u run_longExp.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_sl'$seq_len'_pl'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --period_len $period_len \
      --model_type 'mlp' \
      --d_model 128 \
      --enc_in 7 \
      --train_epochs $train_epochs \
      --patience $patience \
      --itr 1 \
      --batch_size $batch_size \
      --learning_rate $lr_rate \
      --lradj type7 \
      --seed $seed
  done
  done
  done


  data_path_name=ETTm1.csv
  model_id_name=ETTm1
  data_name=ETTm1

  for seed in 2023 2024 2025 2026 2027
  do
  for seq_len in 336 512 720
  do
  for pred_len in 48 96 192 336 512 720
  do
    python -u run_longExp.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_sl'$seq_len'_pl'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --period_len $period_len \
      --model_type 'mlp' \
      --d_model 128 \
      --enc_in 7 \
      --train_epochs $train_epochs \
      --patience $patience \
      --itr 1 \
      --batch_size $batch_size \
      --learning_rate $lr_rate \
      --lradj type7 \
      --seed $seed
  done
  done
  done


  data_path_name=ETTm2.csv
  model_id_name=ETTm2
  data_name=ETTm2

  for seed in 2023 2024 2025 2026 2027
  do
  for seq_len in 336 512 720
  do
  for pred_len in 48 96 192 336 512 720
  do
    python -u run_longExp.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_sl'$seq_len'_pl'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --period_len $period_len \
      --model_type 'mlp' \
      --d_model 128 \
      --enc_in 7 \
      --train_epochs $train_epochs \
      --patience $patience \
      --itr 1 \
      --batch_size $batch_size \
      --learning_rate $lr_rate \
      --lradj type7 \
      --seed $seed
  done
  done
  done
done