# -*-Encoding: utf-8 -*-
################################################################################
#
# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Description: Training and Validation
Authors: Lu,Xinjiang (luxinjiang@baidu.com)
Date:    2022/03/10
"""
import os
import time
import numpy as np
from typing import Callable
import torch
import random
from torch.utils.data import DataLoader
from model import BaselineGruModel
from common import EarlyStopping
from common import adjust_learning_rate
from common import Experiment
from prepare import prep_env
import pandas as pd
import pickle
from datetime import datetime



def get_block_split(total,n_splits=5,ratio=0.8):
    k_fold_size = total//n_splits  # 49days
    train_size = int(ratio * k_fold_size)  # 39
    val_size = k_fold_size - train_size
    return train_size,val_size
    
def val(experiment, model, data_loader, criterion):
    # type: (Experiment, BaselineGruModel, DataLoader, Callable) -> np.array
    """
    Desc:
        Validation function
    Args:
        experiment:
        model:
        data_loader:
        criterion:
    Returns:
        The validation loss
    """
    validation_loss = []
    for i, (batch_x, batch_y) in enumerate(data_loader):
        sample, true = experiment.process_one_batch(model, batch_x, batch_y)
        loss = criterion(sample, true)
        validation_loss.append(loss.item())
    validation_loss = np.average(validation_loss)*(1+3*np.std(validation_loss))
    
    return validation_loss

def train_and_val_cv(experiment, model, model_folder, is_debug=False):
    # type: (Experiment, BaselineGruModel, str, bool) -> tuple[list,list]
    """
    Desc:
        Training and validation
    Args:
        experiment:
        model:
        model_folder: folder name of the model
        is_debug:
    Returns:
        None
    """
    args = experiment.get_args()
    


    path_to_model = os.path.join(args["checkpoints"], model_folder)
    if not os.path.exists(path_to_model):
        os.makedirs(path_to_model)
    model_file_name = 'X.pt'
    feature_scaler_file_name = 'feature_scaler_X'
    target_scaler_file_name = 'target_scaler_X'
    path_to_feature_scaler = os.path.join(path_to_model, feature_scaler_file_name)
    path_to_target_scaler = os.path.join(path_to_model, target_scaler_file_name)
    optimizer = experiment.get_optimizer(model)
    criterion = Experiment.get_criterion()

    epoch_start_time = time.time()

    
    n_splits = 5
    train_days,val_days = get_block_split(args["total_size"],n_splits=n_splits,ratio=0.8)
    cv_train_loss = []
    cv_val_loss = []

    for i  in range(n_splits): 
        print(f"kfold {i}")
        shift_days = i*( train_days+val_days)
        train_data, train_loader = experiment.get_cv_data(flag='train',shift_days=shift_days,train_days=train_days,val_days=train_days)
        val_data, val_loader = experiment.get_cv_data(flag='val',shift_days=shift_days,train_days=train_days,val_days=train_days) 
        train_loss_list = []
        val_loss_list = []
        
        early_stopping = EarlyStopping(patience=args["patience"], verbose=True)

        for epoch in range(args["train_epochs"]):
            iter_count = 0
            train_loss = []
            model.train()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                optimizer.zero_grad()  # PyTorch accumulates gradients on subsequent backward passes
                sample, truth = experiment.process_one_batch(model, batch_x, batch_y)
                loss = criterion(sample, truth)
                train_loss.append(loss.item())
                loss.backward()
                # optimizer.minimize(loss)
                optimizer.step()
            val_loss = val(experiment, model, val_loader, criterion)
            train_loss = np.average(train_loss)
            if is_debug:
                epoch_end_time = time.time()
                print("Epoch: {}, \nTrain Loss: {}, \nValidation Loss: {}".format(epoch, train_loss, val_loss))
                print("Elapsed time for epoch-{}: {}".format(epoch, epoch_end_time - epoch_start_time))
                epoch_start_time = epoch_end_time
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            # Early Stopping if needed
            early_stopping(val_loss, model, path_to_model, model_file_name)
            if early_stopping.early_stop:
                print("Early stopped! ")
                break
            adjust_learning_rate(optimizer, epoch + 1, args)
        cv_train_loss.append(np.min(train_loss_list))
        cv_val_loss.append(np.min(val_loss_list))
        pickle.dump(train_data.feature_scaler_dict,open(path_to_feature_scaler, 'wb'))
        pickle.dump(train_data.target_scaler_dict,open(path_to_target_scaler, 'wb'))

    return cv_train_loss,cv_val_loss

def train_and_val(experiment, model, model_folder, is_debug=False):
    # type: (Experiment, BaselineGruModel, str, bool) -> tuple[list,list]
    """
    Desc:
        Training and validation
    Args:
        experiment:
        model:
        model_folder: folder name of the model
        is_debug:
    Returns:
        None
    """
    args = experiment.get_args()
    train_data, train_loader = experiment.get_global_data(flag='train')
    val_data, val_loader = experiment.get_global_data(flag='val')

    path_to_model = os.path.join(args["checkpoints"], model_folder)
    if not os.path.exists(path_to_model):
        os.makedirs(path_to_model)
    model_file_name = 'X.pt'
    feature_scaler_file_name = 'feature_scaler_X'
    target_scaler_file_name = 'target_scaler_X'
    path_to_feature_scaler = os.path.join(path_to_model, feature_scaler_file_name)
    path_to_target_scaler = os.path.join(path_to_model, target_scaler_file_name)
    early_stopping = EarlyStopping(patience=args["patience"], verbose=True)
    optimizer = experiment.get_optimizer(model)
    criterion = Experiment.get_criterion()

    epoch_start_time = time.time()
    train_loss_list = []
    val_loss_list = []
    for epoch in range(args["train_epochs"]):
        iter_count = 0
        train_loss = []
        model.train()
        for i, (batch_x, batch_y) in enumerate(train_loader):
            iter_count += 1
            optimizer.zero_grad()  # PyTorch accumulates gradients on subsequent backward passes
            sample, truth = experiment.process_one_batch(model, batch_x, batch_y)
            loss = criterion(sample, truth)
            train_loss.append(loss.item())
            loss.backward()
            # optimizer.minimize(loss)
            optimizer.step()
        val_loss = val(experiment, model, val_loader, criterion)
        train_loss = np.average(train_loss)
        if is_debug:
            epoch_end_time = time.time()
            print("Epoch: {}, \nTrain Loss: {}, \nValidation Loss: {}".format(epoch, train_loss, val_loss))
            print("Elapsed time for epoch-{}: {}".format(epoch, epoch_end_time - epoch_start_time))
            epoch_start_time = epoch_end_time
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        # Early Stopping if needed
        early_stopping(val_loss, model, path_to_model, model_file_name)
        if early_stopping.early_stop:
            print("Early stopped! ")
            break
        adjust_learning_rate(optimizer, epoch + 1, args)
    pickle.dump(train_data.feature_scaler_dict,open(path_to_feature_scaler, 'wb'))
    pickle.dump(train_data.target_scaler_dict,open(path_to_target_scaler, 'wb'))

    return train_loss_list,val_loss_list

