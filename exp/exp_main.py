from contextlib import nullcontext
import shutil
import os
import time
#from unittest import loader
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from data_provider.data_factory import TimeSeriesDataset
from models import DLinear, ModernTCN, PatchTST, SparseTSF, iTransformer, FrNet
from utils.learning_rates import adjust_learning_rate
from utils.tools import EarlyStopping
from utils.metrics import metric



class Exp_Main:
    '''
    Main experiment class that handles training and testing of the model. It inherits
    from Exp_Basic and implements the necessary methods for building the model, 
    getting data, selecting optimizer and criterion, training, and testing.
    '''
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()

        if self.args.d_is_training:
            self.train_loader = self._get_data(flag='train')
            self.vali_loader = self._get_data(flag='val')
            self.test_loader = self._get_data(flag='test')
        else:
            self.test_loader = self._get_data(flag='test')

        self.model = self._build_model().to(self.device)


    def _acquire_device(self):
        '''
        This method checks if GPU is available and sets the device accordingly. 
        If GPU is available and use_gpu flag is set, it sets the device to the 
        specified GPU. If use_multi GPU flag is set, it allows the use of multiple 
        GPUs. If GPU is not available, it sets the device to CPU.
        '''
        if self.args.d_use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.d_gpu) if not self.args.d_use_multi_gpu else self.args.d_devices
            device = torch.device('cuda:{}'.format(self.args.d_gpu))
            print('Use GPU: cuda:{}'.format(self.args.d_gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _build_model(self):
        '''
        This method builds the model based on the specified architecture in the arguments.
        It uses a dictionary to map model names to their corresponding classes and 
        initializes the model with the given arguments.
        
        Add your model to the dictionary if you want to use it in the experiment. 
        Make sure to import the model at the top of this file.
        '''
        
        model_dict = {
            'DLinear': DLinear,
            'PatchTST': PatchTST,
            'SparseTSF': SparseTSF,
            'iTransformer': iTransformer,
            'ModernTCN': ModernTCN,
            'FrNet': FrNet,
        }
        
        model = model_dict[self.args.d_model].Model(self.args).float()

        # If multiple GPUs are to be used and GPU is available, 
        # wrap the model with DataParallel for multi-GPU training.
        if self.args.d_use_multi_gpu and self.args.d_use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        
        return model

    def _get_data(self, flag):
 
        timeenc = 0 if self.args.d_embed != 'timeF' else 1

        # Loader configuration
        flag_list = {
            "train": {"shuffle": True,
                    "drop_last": True, 
                    "stride": self.args.d_stride, 
                    "batch_size": self.args.d_batch_size},

            "val": {"shuffle": False, 
                    "drop_last": False, 
                    "stride": self.args.d_pred_len, 
                    "batch_size": self.args.d_batch_size},

            "test": {"shuffle": False, 
                     "drop_last": False, 
                     "stride": self.args.d_pred_len,
                     "batch_size": 1},
                }

        dataset = TimeSeriesDataset(
            args = self.args,
            timeenc=timeenc,
            flag=flag,
            stride=flag_list[flag]["stride"]
        )

        print(f"{flag} Dataset: ({len(dataset)}, {self.args.d_seq_len}, {self.args.d_in_features})")

        loader = DataLoader(
            dataset,
            batch_size=flag_list[flag]["batch_size"],
            shuffle=flag_list[flag]["shuffle"],
            num_workers=self.args.d_num_workers,
            drop_last=flag_list[flag]["drop_last"],
            pin_memory=True
        )

        return loader

    def _select_optimizer(self):
        '''
        Selects the optimizer for training the model. Currently, it uses Adam optimizer
        with the learning rate specified in the arguments. 
        '''
        
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.d_learning_rate)
        
        return model_optim

    def _select_criterion(self):
        '''
        Selects the loss function for training the model. Currently, it uses MSELoss
        with the loss specified in the arguments. Add more loss functions to the 
        losses_dict if needed.
        '''
        
        losses_dict = {
            'mae': nn.L1Loss(),
            'mse': nn.MSELoss(),
            'smooth': nn.SmoothL1Loss(),
        }
        criterion = losses_dict.get(self.args.d_loss, nn.MSELoss())
        
        return criterion


    def _get_inputs(self, x, y, x_mark, y_mark):
        '''
        This method prepares the inputs for the model during training and validation. 
        It moves the input data (batch_x, batch_y) to the appropriate device (CPU or GPU) 
        for computation. For certain datasets like PEMS and Solar, the time features 
        (batch_x_mark and batch_y_mark) are not used, so they are set to None. 
        For other datasets, they are moved to the appropriate device for computation. 
        The decoder input is created by concatenating the known part of the target 
        sequence (label_len) with a zero tensor for the unknown part (pred_len). 
        This is used as input to the decoder during training and validation.
        '''
        # Move the input data (batch_x, batch_y) to the appropriate device (CPU or GPU)
        # for computation.
        batch_x = x.float().to(self.device)
        batch_y = y.float().to(self.device)
        
        # For certain datasets like PEMS and Solar, the time features (batch_x_mark and batch_y_mark)
        # are not used, so they are set to None. For other datasets, they are moved to the appropriate device for computation.
        if 'PEMS' in self.args.d_data or 'Solar' in self.args.d_data:
            batch_x_mark = None
            batch_y_mark = None
        else:
            batch_x_mark = x_mark.float().to(self.device)
            batch_y_mark = y_mark.float().to(self.device)

        # Decoder input is created by concatenating the known part of the target sequence
        # (label_len) with a zero tensor for the unknown part (pred_len). 
        # This is used as input to the decoder during training and validation.
        dec_inp = torch.zeros_like(y[:, -self.args.d_pred_len:, :]).float()
        dec_inp = torch.cat([y[:, :self.args.d_label_len, :], dec_inp], dim=1).float().to(self.device)

        return batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp

            

    def vali(self, vali_loader, criterion):
        '''
        This method evaluates the model on the validation set. It iterates 
        through the validation dataloader, makes predictions using the model,
        and calculates the loss using the specified criterion. The average loss
        over the validation set is returned. The model is set to evaluation mode
        during this process and then switched back to training mode at the end.
        '''
        
        # List to store the loss for each batch in the validation set.
        total_loss = []
        
        # Set the model to evaluation mode to disable dropout and batch normalization layers.
        self.model.eval()
        
        with torch.no_grad():
            for _, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                
                batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp = self._get_inputs(batch_x, batch_y, batch_x_mark, batch_y_mark)

                # Use automatic mixed precision for inference if enabled. 
                # This can speed up inference on compatible hardware.
                amp_context = torch.cuda.amp.autocast() if self.args.d_use_amp else nullcontext()

                input_type = self.args.d_model_input_type.lower()

                if input_type == "x_only":
                    model_args = (batch_x,)
                elif input_type == "x_mark_incl":
                    model_args = (batch_x, batch_x_mark)
                else:
                    model_args = (batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    
                with amp_context:
                    outputs = self.model(*model_args)
                
                if input_type not in ["x_only", "x_mark_incl"] and self.args.d_output_attention:
                    outputs = outputs[0]

                f_dim = -1 if self.args.d_forecast_type == 'MS' else 0
                outputs = outputs[:, -self.args.d_pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.d_pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                loss = criterion(pred, true)
                total_loss.append(loss)

        total_loss = np.average(total_loss)

        self.model.train()

        return total_loss

    def train(self, setting):
        '''
        This method handles the training loop for the model. It first gets the training,
        validation, and test data loaders. It then sets up the checkpoint directory, 
        initializes the optimizer, criterion, and learning rate scheduler. The training 
        loop iterates over the specified number of epochs, where in each epoch it iterates
        over the training data, makes predictions, calculates the loss, and updates the 
        model parameters.
        '''

        # Set up the checkpoint directory for saving the best model during training. The path
        # is constructed using the base checkpoints directory and the specific setting for 
        # this experiment. If the directory does not exist it is created.
        path = os.path.join(self.args.d_checkpoint_path, self.args.d_setting)

        time_now = time.time()

        # Calculate the number of training steps per epoch based on the length of the training 
        # data loader.
        train_steps = len(self.train_loader)
        early_stopping = EarlyStopping(patience=self.args.d_patience, verbose=True)

        # Initialize the optimizer, criterion, and learning rate scheduler for training. 
        # The optimizer is selected using the _select_optimizer method, and 
        # the criterion is selected using the _select_criterion method.
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # If automatic mixed precision is enabled, initialize the GradScaler for scaling 
        # the loss during backpropagation.
        if self.args.d_use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # Set up the learning rate scheduler. If the learning rate adjustment method is 'TST',
        # it uses the OneCycleLR scheduler which adjusts the learning rate according to an annealing
        # strategy. The scheduler is configured with the number of steps per epoch, percentage 
        # of the cycle for increasing the learning rate, total number of epochs, and the maximum
        # learning rate specified in the arguments.
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.d_pct_start,
                                            epochs=self.args.d_train_epochs,
                                            max_lr=self.args.d_learning_rate)

        for epoch in range(self.args.d_train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(self.train_loader):
                iter_count += 1

                # Clear the gradients of the model parameters before backpropagation. This is 
                # necessary to prevent the accumulation of gradients from multiple batches, 
                # which can lead to incorrect updates of the model parameters.
                model_optim.zero_grad()

                batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp = self._get_inputs(batch_x, batch_y, batch_x_mark, batch_y_mark)

                # Use automatic mixed precision for inference if enabled. 
                # This can speed up inference on compatible hardware.
                amp_context = torch.cuda.amp.autocast() if self.args.d_use_amp else nullcontext()

                input_type = self.args.d_model_input_type.lower()

                if input_type == "x_only":
                    model_args = (batch_x,)
                elif input_type == "x_mark_incl":
                    model_args = (batch_x, batch_x_mark)
                else:
                    model_args = (batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    
                with amp_context:
                    outputs = self.model(*model_args)
                
                if input_type not in ["x_only", "x_mark_incl"] and self.args.d_output_attention:
                    outputs = outputs[0]

                f_dim = -1 if self.args.d_forecast_type == 'MS' else 0
                outputs = outputs[:, -self.args.d_pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.d_pred_len:, f_dim:].to(self.device)

                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                # Print training progress every 100 iterations. It calculates the speed 
                # of training and the estimated time left for the current epoch based 
                # on the number of iterations completed and the total number of iterations
                # in the epoch. The loss for the current batch is also printed.
                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.d_train_epochs - epoch) * train_steps - i)
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # Backpropagation and optimization step. If automatic mixed precision 
                # is enabled, the loss is scaled before backpropagation to prevent 
                # underflow issues. The optimizer step is also performed using the scaled
                # gradients, and the scaler is updated after the step.
                # If automatic mixed precision is not enabled, the loss is backpropagated
                # and the optimizer step is performed in the standard way.
                if self.args.d_use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                # Adjust the learning rate using the scheduler. If the learning rate adjustment method is 'TST',
                # the learning rate is adjusted according to the OneCycleLR scheduler after each batch.
                if self.args.d_lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(self.vali_loader, criterion)
            test_loss = self.vali(self.test_loader, criterion)

            print("\nEpoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            
            # Check for early stopping based on the validation loss. The EarlyStopping class 
            # is used to monitor the validation loss and save the best model. If the 
            # validation loss does not improve for a specified number of epochs (patience),
            # the training loop will be stopped early to prevent overfitting.
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # Adjust the learning rate using the scheduler if the learning rate adjustment method is not 'TST'.
            # If the method is 'TST', the learning rate is already adjusted after each batch, 
            # so it is not adjusted again here.    
            if self.args.d_lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        # After training is complete, the best model is loaded from the checkpoint directory for testing.
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path, map_location="cpu"))


        return self.model

    def test(self, setting):
        '''
        This method handles the testing of the model. It first gets the test data loader, 
        and if the test flag is set, it loads the best model from the checkpoint directory. 
        It then iterates through the test data, makes predictions using the model, and 
        calculates various metrics (MAE, MSE, RMSE, MAPE, MSPE, RSE, Correlation) to evaluate
        the performance of the model on the test set. The results are saved to a CSV 
        file and optionally plotted and saved as numpy arrays.
        '''

        if not self.args.d_is_training:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location="cpu"))

        preds = []
        trues = []
        #inputx = []

        # Inputs for FlopCounts for fvncore
        # profile_x = None
        # profile_x_mark = None
        # profile_y_mark = None
        # profile_dec_inp = None

        self.model.eval()
        #if self.args.d_call_structural_reparam and hasattr(self.model, 'structural_reparam'):
        #    self.model.structural_reparam()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(self.test_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp = self._get_inputs(batch_x, batch_y, batch_x_mark, batch_y_mark)

                # # Creating profiles for Fvncore
                # if i==0:
                #     profile_x = torch.randn(batch_x.shape).to(self.device)
                #     profile_x_mark = torch.randn(batch_x_mark.shape).to(self.device)
                #     profile_y_mark = torch.randn(batch_y_mark.shape).to(self.device)
                #     profile_dec_inp = torch.randn(dec_inp.shape).to(self.device)

                input_type = self.args.d_model_input_type.lower()

                if input_type == "x_only":
                    model_args = (batch_x,)
                elif input_type == "x_mark_incl":
                    model_args = (batch_x, batch_x_mark)
                else:
                    model_args = (batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                outputs = self.model(*model_args)

                if input_type not in ["x_only", "x_mark_incl"] and self.args.d_output_attention:
                    outputs = outputs[0]


                f_dim = -1 if self.args.d_forecast_type == 'MS' else 0
                outputs = outputs[:, -self.args.d_pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.d_pred_len:, f_dim:].to(self.device)

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                # if self.args.d_inverse:
                #     shape = outputs.shape
                #     outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                #     batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                #inputx.append(batch_x.detach().cpu().numpy())

                # # If Plotting Results is enabled: 0=False, 1=True
                # if self.args.plot_results:
                #     if i % 20 == 0:
                #         input = batch_x.detach().cpu().numpy()
                #         # if test_data.scale and self.args.inverse:
                #         #     shape = input.shape
                #         #     input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                #         gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                #         pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                #         visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)


        total_test_len = self.test_loader.dataset._data_len()
        valid_pred_len = total_test_len - self.args.d_seq_len

        num_windows = preds.shape[0]
        pred_len = preds.shape[1]
        
        # flatten all windows except the last
        flat_preds = preds[:-1].reshape(-1, preds.shape[-1])
        flat_trues = trues[:-1].reshape(-1, trues.shape[-1])
        
        current_len = flat_preds.shape[0]
        remaining = valid_pred_len - current_len

        if remaining > 0:
            # need extra values from last window
            flat_preds = np.concatenate([flat_preds, preds[-1][-remaining:]], axis=0)
            flat_trues = np.concatenate([flat_trues, trues[-1][-remaining:]], axis=0)
        elif remaining < 0:
            # too many values, need to trim tail
            flat_preds = flat_preds[:valid_pred_len]
            flat_trues = flat_trues[:valid_pred_len]


        #Calculate Metrics.
        mae, mse, rmse, mape, mspe, rse, corr, r2 = metric(flat_preds, flat_trues)


        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        

        with open(f"Result_{self.args.d_model}.csv", 'a') as f:
            f.write(f"{setting}-mse:{mse}-mae:{mae}\n")


        return flat_preds, flat_trues
    

        # #Calculate FLOPS and Params.
        # if self.args.model in ("iTransformer"):
        #     flops = FlopCountAnalysis(self.model, inputs=(profile_x, profile_x_mark, profile_dec_inp, profile_y_mark)).total()
        # else:
        #     flops = FlopCountAnalysis(self.model, profile_x).total()

        # params = sum(p.numel() for p in self.model.parameters())

        # print(f"MACS:{flops}, Params: {params}")

        # if self.args.delete_checkpoints:
        #     checkpoint_path = os.path.join('./checkpoints/' + setting)
        #     if os.path.exists(checkpoint_path):
        #         shutil.rmtree(checkpoint_path)

                # # If Saving Results is enabled: 0=False, 1=True
        # if self.args.save_results:
        #     folder_path = './results/' + setting + '/'
        #     if not os.path.exists(folder_path):
        #         os.makedirs(folder_path)
        #     np.save(folder_path + 'pred.npy', preds)
        #     np.save(folder_path + 'true.npy', trues)