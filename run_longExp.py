import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np
from utils.str2bool import str2bool
from utils.tools import check_and_prepare_dirs, save_results, inverse_transform
#from utils.plotting import plot_results

parser = argparse.ArgumentParser(description='Baseline LTSF Models')

# Arguments that start as "--d_" are DEFAULT arguments to build and load datasets,
# setup forecasting, optimization and gpu settings.
# 
# Use "--m_" to set arguments for your own model, which will be captured automatically.
# This will allow different models to use same arguments if needed, without altercation.

# basic config
parser.add_argument('--d_seed', type=int, default=2021, help='random seed')
parser.add_argument('--d_is_training', type=int, required=True, default=1, help='status')
#parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')


# data configuration
parser.add_argument('--d_data', type=str, required=False, default='Custom')
parser.add_argument('--d_root_path', type=str, default='./dataset/', help='root path of the data file')
parser.add_argument('--d_data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--d_forecast_type', type=str, default='S',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate,' \
                    ' S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--d_target', type=str, default=-1, help='target feature in S or MS task')
parser.add_argument('--d_embed', type=str, default='timeF',
                     help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--d_freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, ' \
                    'd:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--d_checkpoint_path', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--d_stride', type=int, default=10)
parser.add_argument('--d_inverse_transform', action="store_true", default=False)

# model name and forecasting details
parser.add_argument('--d_model', type=str, required=True, default='DLinear', help='model name')
parser.add_argument('--d_model_input_type', type=str, required=True, default="x_only")
parser.add_argument('--d_seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--d_label_len', type=int, default=48, help='start token length for decoder if being used')
parser.add_argument('--d_pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--d_in_features', type=int, default=7, help='number of features or variates')
parser.add_argument('--d_out_features', type=int, default=7, help='number of features or variates')

#
parser.add_argument('--d_output_attention', action='store_true', default=False)

# DLinear
#parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0') # Used by PatchTST too.

# PatchTST
# parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
# parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
# parser.add_argument('--patch_len', type=int, default=16, help='patch length')
# parser.add_argument('--stride', type=int, default=8, help='stride')
# parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
# parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
# parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
# parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
# parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
# parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')

# SparseTSF
# parser.add_argument('--period_len', type=int, default=24, help='period length')
# parser.add_argument('--model_type', default='linear', help='model type: linear/mlp')


# iTransformer
# parser.add_argument('--exp_name', type=str, required=False, default='MTSF',
#                     help='experiemnt name, options:[MTSF, partial_train]')
# parser.add_argument('--channel_independence', type=bool, default=False, help='whether to use channel_independence mechanism')

# parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')
# parser.add_argument('--target_root_path', type=str, default='./data/electricity/', help='root path of the data file')
# parser.add_argument('--target_data_path', type=str, default='electricity.csv', help='data file')
# parser.add_argument('--efficient_training', type=bool, default=False, help='whether to use efficient_training (exp_name should be partial train)') # See Figure 8 of our paper for the detail
# parser.add_argument('--use_norm', type=int, default=True, help='use norm and denorm')
# parser.add_argument('--partial_start_index', type=int, default=0, help='the start index of variates for partial training, '
#                                                                            'you can select [partial_start_index, min(enc_in + partial_start_index, N)]')
#FRNet
# parser.add_argument('--pred_head_type', type=str, default='linear', help='linear or truncation')
# parser.add_argument('--aggregation_type', type=str, default='linear', help='linear or avg')
# parser.add_argument('--channel_attention', type=int, default=0, help='True 1 or False 0')
# parser.add_argument('--global_freq_pred', type=int, default=1, help='True 1 or False 0')
parser.add_argument('--m_period_list', type=int, nargs='+', default=1, help='period_list') 
# parser.add_argument('--emb', type=int, default=64, help='patch embedding size')


#ModernTCN
# parser.add_argument('--stem_ratio', type=int, default=6, help='stem ratio')
# parser.add_argument('--downsample_ratio', type=int, default=2, help='downsample_ratio')
# parser.add_argument('--ffn_ratio', type=int, default=2, help='ffn_ratio')
# parser.add_argument('--patch_size', type=int, default=16, help='the patch size')
# parser.add_argument('--patch_stride', type=int, default=8, help='the patch stride')
# parser.add_argument('--num_blocks', nargs='+',type=int, default=[1,1,1,1], help='num_blocks in each stage')
# parser.add_argument('--large_size', nargs='+',type=int, default=[31,29,27,13], help='big kernel size')
# parser.add_argument('--small_size', nargs='+',type=int, default=[5,5,5,5], help='small kernel size for structral reparam')
# parser.add_argument('--dims', nargs='+',type=int, default=[256,256,256,256], help='dmodels in each stage')
# parser.add_argument('--dw_dims', nargs='+',type=int, default=[256,256,256,256])
# parser.add_argument('--small_kernel_merged', type=str2bool, default=False, help='small_kernel has already merged or not')
# parser.add_argument('--d_call_structural_reparam', type=bool, default=False, help='structural_reparam after training')
# parser.add_argument('--use_multi_scale', type=str2bool, default=True, help='use_multi_scale fusion')


# Formers 
# parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
# parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
# parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
# parser.add_argument('--c_out', type=int, default=7, help='output size')
# parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
# parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
# parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
# parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
# parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
# parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
# parser.add_argument('--factor', type=int, default=1, help='attn factor')
# parser.add_argument('--distil', action='store_false',
#                     help='whether to use distilling in encoder, using this argument means not using distilling',
#                     default=True)
# parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
 
# parser.add_argument('--activation', type=str, default='gelu', help='activation')
# parser.add_argument('--output_attention', action='store_true', default=False, help='whether to output attention in ecoder')
# parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')


# optimization
parser.add_argument('--d_num_workers', type=int, default=10, help='data loader num workers')
#parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--d_train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--d_batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--d_patience', type=int, default=6, help='early stopping patience')
parser.add_argument('--d_learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--d_des', type=str, default='test', help='exp description')
parser.add_argument('--d_loss', type=str, default='mse', help='loss function')
parser.add_argument('--d_lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--d_pct_start', type=float, default=0.3, help='pct_start')
parser.add_argument('--d_use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--d_use_gpu', action="store_true", default=False, help='use gpu')
parser.add_argument('--d_gpu', type=int, default=0, help='gpu')
parser.add_argument('--d_use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--d_devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
#parser.add_argument('--d_test_flop', action='store_true', default=False, help='See utils/tools for usage')

args, unknown = parser.parse_known_args()

# --------------------------------
# Evaluate and raise error if unkown arguments are not in pairs.
# --------------------------------
for i in range(0, len(unknown), 2):
    key = unknown[i]

    # check key format
    if not key.startswith("--"):
        raise ValueError(
            f"Invalid key '{key}'. Expected format '--key value'."
        )

    # check value exists
    if i + 1 >= len(unknown):
        raise ValueError(
            f"Missing value for argument '{key}'."
        )

# --------------------------------
# Add the unkown arguments into known args dictionary.
# --------------------------------
extra_args = {}

# Function to convert the unkown arguments into bool, int, float or str.
def parse_value(v):
    v_lower = v.lower()

    # handle boolean
    if v_lower == "true":
        return True
    if v_lower == "false":
        return False

    # try integer
    try:
        return int(v)
    except ValueError:
        pass

    # try float
    try:
        return float(v)
    except ValueError:
        pass

    # fallback to string
    return v

# Add extra arguments into args dictionary.
for i in range(0, len(unknown), 2):
    key = unknown[i].lstrip('-')
    value = parse_value(unknown[i + 1])
    extra_args[key] = value

for k, v in extra_args.items():
    setattr(args, k, v)
     

# --------------------------------
# Set the seed value.
# --------------------------------
fix_seed = args.d_seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


# --------------------------------
# Set the GPU settings.
# --------------------------------
args.d_use_gpu = True if torch.cuda.is_available() and args.d_use_gpu else False

# If using multiple GPUs, set up the device ids and the primary gpu.
if args.d_use_gpu and args.d_use_multi_gpu:
    args.d_devices = args.devices.replace(' ', '')
    device_ids = args.d_devices.split(',')
    args.d_device_ids = [int(id_) for id_ in device_ids]
    args.d_gpu = args.device_ids[0]

# --------------------------------
# Print the arguments for the experiment. 
# This is useful for logging and reproducibility.
# --------------------------------
print('Args in experiment:')
print(args)

setting = '{}-{}-ft{}-sl{}-pl{}-seed{}'.format(
    args.d_model,
    args.d_data,
    args.d_forecast_type,
    args.d_seq_len,
    args.d_pred_len,
    args.d_seed)

args.d_setting = setting

# --------------------------------
# Prepare directories.
# --------------------------------
check_and_prepare_dirs(args)

# --------------------------------
# RUN EXPERIMENT.
# --------------------------------
exp = Exp_Main(args)

# If training is enabled.
if args.d_is_training:
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(args.d_setting))
        exp.train(setting)

print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(args.d_setting))
preds, trues = exp.test(args.d_setting)

# If output needs to be inverse transformed.
if args.d_inverse_transform:
     preds = inverse_transform(args, preds)
     trues = inverse_transform(args, trues)

# --------------------------------
# Save and plot results.
# --------------------------------
save_results(args, preds, trues)
#plot_results(args, preds, trues, zoom_to=400)

torch.cuda.empty_cache()
