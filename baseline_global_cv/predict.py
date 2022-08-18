# -*-Encoding: utf-8 -*-
################################################################################
#
# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Description: A demo of the forecasting method
Authors: Lu,Xinjiang (luxinjiang@baidu.com)
Date:    2022/04/18
"""
import os
import time
import numpy as np
import torch
from model import BaselineGruModel
from common import Experiment
from wind_turbine_data import feature_eng,inverse_sigmoid,sigmoid
from test_data import TestData
import pickle

def forecast_one(experiment, test_turbines):
    # type: (Experiment, TestData) -> np.ndarray
    """
    Desc:
        Forecasting the power of one turbine
    Args:
        experiment:
        test_turbines:
        train_data:
    Returns:
        Prediction for one turbine
    """
    args = experiment.get_args()
    # load models and scalers
    tid = args["turbine_id"]
    model = experiment.model.to(args['device'])
    model_dir = '{}_t{}_i{}_o{}_ls{}_train{}_val{}'.format(
        args["filename"], args["task"], args["input_len"], args["output_len"], args["lstm_layer"],
        args["train_size"], args["val_size"]
    )
    model_file_name = 'model_X.pt'
    feature_scaler_file_name = 'feature_scaler_X'
    target_scaler_file_name = 'target_scaler_X'
    path_to_model = os.path.join(args["checkpoints"], model_dir, model_file_name)
    path_to_feature_scaler = os.path.join(args["checkpoints"], model_dir, feature_scaler_file_name)
    path_to_target_scaler = os.path.join(args["checkpoints"], model_dir, target_scaler_file_name)
    model.load_state_dict(torch.load(path_to_model))
    feature_scaler_dict = pickle.load(open(path_to_feature_scaler, 'rb'))
    target_scaler_dict = pickle.load(open(path_to_target_scaler, 'rb'))
    g_feature_scaler = feature_scaler_dict['_GLOBAL']
    g_target_scaler = target_scaler_dict['_GLOBAL']
    feature_scaler = feature_scaler_dict.get(tid,g_feature_scaler)
    target_scaler = target_scaler_dict.get(tid,g_target_scaler)
    test_x, _ = test_turbines.get_turbine(tid)
    # feature enginieering
    test_x = feature_eng(test_x)
    test_x = feature_scaler.transform(test_x)
    last_observ = test_x[-args["input_len"]:]
    seq_x = torch.tensor(last_observ)
    sample_x = seq_x.reshape(-1, seq_x.shape[-2], seq_x.shape[-1])
    prediction = experiment.inference_one_sample(model, sample_x)
    prediction = target_scaler.inverse_transform(prediction.cpu().detach().numpy()[0])
    #prediction = sigmoid(prediction)
    return prediction#.cpu().detach().numpy()


def forecast(settings):
    # type: (dict) -> np.ndarray
    """
    Desc:
        Forecasting the wind power in a naive distributed manner
    Args:
        settings:
    Returns:
        The predictions
    """
    start_time = time.time()
    predictions = []
    settings["turbine_id"] = 0
    exp = Experiment(settings)
    # train_data = Experiment.train_data
    if settings["is_debug"]:
        end_train_data_get_time = time.time()
        print("Load train data in {} secs".format(end_train_data_get_time - start_time))
        start_time = end_train_data_get_time
    test_x = Experiment.get_test_x(settings)
    if settings["is_debug"]:
        end_test_x_get_time = time.time()
        print("Get test x in {} secs".format(end_test_x_get_time - start_time))
        start_time = end_test_x_get_time
    for i in range(settings["capacity"]):
        settings["turbine_id"] = i
        # print('\n>>>>>>> Testing Turbine {:3d} >>>>>>>>>>>>>>>>>>>>>>>>>>\n'.format(i))
        prediction = forecast_one(exp, test_x)
        torch.cuda.empty_cache()
        predictions.append(prediction)
        if settings["is_debug"] and (i + 1) % 10 == 0:
            end_time = time.time()
            print("\nElapsed time for predicting 10 turbines is {} secs".format(end_time - start_time))
            start_time = end_time
    return np.array(predictions)
