# -*-Encoding: utf-8 -*-
################################################################################
#
# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Description: Common utilities for Wind Power Forecasting
Authors: Lu,Xinjiang (luxinjiang@baidu.com)
Date:    2022/03/10
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import BaselineGruModel
from wind_turbine_data import GlobalWindTurbineDataset
from test_data import TestData

# class Custom_MSE(nn.Module):
#     def __init__(self):
#         super(Custom_MSE, self).__init__();

#     def forward(self, predictions, target):
#         square_difference = torch.square(predictions - target)
#         mask = torch.gt(target,-0.0001)
#         masked = torch.multiply( square_difference,  mask)          
#         loss_value = torch.mean(masked)
#         return loss_value

def Custom_MSE(predictions, target):   
    # args= torch.argwhere(target >=0)
    # corr_rmse = torch.sum(torch.sqrt(torch.mean((predictions[args[:,0],args[:,1]]- target[args[:,0],args[:,1]])**2)))
    corr_rmse = nn.functional.smooth_l1_loss(predictions, target,beta=1) # 0.5 good  #1.5 42.30
    return corr_rmse


def adjust_learning_rate(optimizer, epoch, args):
    # type: (torch.optim.Adam, int, dict) -> None
    """
    Desc:
        Adjust learning rate
    Args:
        optimizer:
        epoch:
        args:
    Returns:
        None
    """
    # lr = args.lr * (0.2 ** (epoch // 2))
    lr_adjust = {}
    if args["lr_adjust"] == 'type1':
        # learning_rate = 0.5^{epoch-1}
        lr_adjust = {epoch: args["lr"] * (0.50 ** (epoch - 1))}
    elif args["lr_adjust"] == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust:
        lr = lr_adjust[epoch]
        optimizer.lr=lr


class EarlyStopping(object):
    """
    Desc:
        EarlyStopping
    """
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_model = False

    def save_checkpoint(self, val_loss, model, path, tid):
        # type: (nn.MSELoss, BaselineGruModel, str, str) -> None
        """
        Desc:
            Save current checkpoint
        Args:
            val_loss: the validation loss
            model: the model
            path: the path to be saved
            tid: turbine ID
        Returns:
            None
        """
        self.best_model = True
        self.val_loss_min = val_loss
        torch.save(model.state_dict(), path + '/' + 'model_' + tid)

    def __call__(self, val_loss, model, path, tid):
        # type: (nn.MSELoss, BaselineGruModel, str, str) -> None
        """
        Desc:
            __call__
        Args:
            val_loss: the validation loss
            model: the model
            path: the path to be saved
            tid: turbine ID
        Returns:
            None
        """
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, tid)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.best_model = False
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.update_hidden = True
            self.save_checkpoint(val_loss, model, path, tid)
            self.counter = 0


class Experiment(object):
    """
    Desc:
        The experiment to train, validate and test a model
    """
    def __init__(self, args):
        # type: (dict) -> None
        """
        Desc:
            __init__
        Args:
            args: the arguments to initialize the experimental environment
        """
        self.args = args
        self.model = BaselineGruModel(self.args)

    def get_args(self):
        # type: () -> dict
        """
        Desc:
            Get the arguments
        Returns:
            A dict
        """
        return self.args

    def get_global_data(self, flag):
        # type: (str) -> (GlobalWindTurbineDataset, DataLoader)
        """
        Desc:
            get_data
        Args:
            flag: train or test
        Returns:
            A dataset and a dataloader
        """
        if flag == 'test':
            shuffle_flag = False
            drop_last = True
        else:
            shuffle_flag = True
            drop_last = True

        turbine_list = [i for i in range(self.args["capacity"])]

        data_set = GlobalWindTurbineDataset(
            data_path=self.args["data_path"],
            filename=self.args["filename"],
            flag=flag,
            size=[self.args["input_len"], self.args["output_len"]],
            task=self.args["task"],
            target=self.args["target"],
            start_col=self.args["start_col"],
            turbine_id_list=turbine_list,
            day_len=self.args["day_len"],
            train_days=self.args["train_size"],
            val_days=self.args["val_size"],
            total_days=self.args["total_size"]
        )
        data_loader = DataLoader(
            data_set,
            batch_size=self.args["batch_size"],
            shuffle=shuffle_flag,
            num_workers=self.args["num_workers"],
            drop_last=drop_last
        )
        return data_set, data_loader
    
    def get_cv_data(self, flag,shift_days,train_days,val_days):
        # type: (str) -> (GlobalWindTurbineDataset, DataLoader)
        """
        Desc:
            get_data
        Args:
            flag: train or test
        Returns:
            A dataset and a dataloader
        """
        if flag == 'test':
            shuffle_flag = False
            drop_last = True
        else:
            shuffle_flag = True
            drop_last = True

        turbine_list = [i for i in range(self.args["capacity"])]

        data_set = GlobalWindTurbineDataset(
            data_path=self.args["data_path"],
            filename=self.args["filename"],
            flag=flag,
            size=[self.args["input_len"], self.args["output_len"]],
            task=self.args["task"],
            target=self.args["target"],
            start_col=self.args["start_col"],
            turbine_id_list=turbine_list,
            day_len=self.args["day_len"],
            train_days=train_days,
            val_days=val_days,
            total_days=self.args["total_size"],
            shift_days = shift_days
        )
        data_loader = DataLoader(
            data_set,
            batch_size=self.args["batch_size"],
            shuffle=shuffle_flag,
            num_workers=self.args["num_workers"],
            drop_last=drop_last
        )
        return data_set, data_loader    


    def get_optimizer(self, model):
        # type: (BaselineGruModel) -> torch.optim.Adam
        """
        Desc:
            Get the optimizer
        Returns:
            An optimizer
        """
        model_optim = torch.optim.Adam(params=model.parameters(),
                                            lr=self.args["lr"],
                                      )
        return model_optim

    @staticmethod
    def get_criterion():
        # type: () -> nn.MSELoss
        """
        Desc:
            Use the mse loss as the criterion
        Returns:
            MSE loss
        """
        #criterion = nn.MSELoss(reduction='mean')
        criterion = Custom_MSE
        return criterion

    def process_one_batch(self, model, batch_x, batch_y):
        # type: (BaselineGruModel, torch.tensor, torch.tensor) -> (torch.tensor, torch.tensor)
        """
        Desc:
            Process a batch
        Args:
            model:
            batch_x:
            batch_y:
        Returns:
            prediction and ground truth
        """
        batch_x = batch_x.type(torch.float).to(self.args['device'])
        batch_y = batch_y.type(torch.float).to(self.args['device'])
        sample = self.model.to(self.args['device'])(batch_x)
        #
        f_dim = -1 if self.args["task"] == 'MS' else 0
        #
        batch_y = batch_y[:, -self.args["output_len"]:, f_dim:].type(torch.float)
        sample = sample[..., :, f_dim:].type(torch.float)
        return sample, batch_y

    @staticmethod
    def get_test_x(args):
        # type: (dict) -> TestData
        """
        Desc:
            Obtain the input sequence for testing
        Args:
            args:
        Returns:
            Normalized input sequences and training data
        """
        test_x = TestData(path_to_data=args["path_to_test_x"], farm_capacity=args["capacity"])
        return test_x

    def inference_one_sample(self, model, sample_x):
        # type: (BaselineGruModel, torch.tensor) -> torch.tensor
        """
        Desc:
            Inference one sample
        Args:
            model:
            sample_x:
        Returns:
            Predicted sequence with sample_x as input
        """
        x = sample_x.type(torch.float).to(self.args['device'])
        prediction = model(x)
        f_dim = -1 if self.args["task"] == 'MS' else 0
        return prediction[..., :, f_dim:].type(torch.float)
