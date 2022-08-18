# -*-Encoding: utf-8 -*-
################################################################################
#
# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Description: Prepare the experimental settings
"""
import torch


def prep_env():
    # type: () -> dict
    """
    Desc:
        Prepare the experimental settings
    Returns:
        The initialized arguments
    """
    settings = {
        "path_to_test_x": "./data/sdwpf_baidukddcup2022_test_toy/test_x",
        "path_to_test_y": "./data/sdwpf_baidukddcup2022_test_toy/test_y",
        "data_path": "./data",
        "filename": "wtbdata_245days.csv",
        "task": "MS",
        "target": "Patv",
        "checkpoints": "BR_checkpoints",
        "input_len": 32,
        "output_len": 288,
        "start_col": 3,
        "in_var": 3,
        "out_var": 1,
        "day_len": 144,
        "train_size": 214,
        "val_size": 31,
        "total_size": 245,
        "lstm_layer": 2,
        "dropout": 0.05,
        "num_workers": 0,
        "train_epochs": 50,
        "batch_size": 613,
        "patience": 3,
        "lr": 0.00085270843141,
        "lr_adjust": "type1",
        "gpu": 0,
        "capacity": 134,
        "turbine_id": 0,
        "pred_file": "predict.py",
        "framework": "pytorch",
        "is_debug": True,
        "hidden_dim":38,
    }
    ###
    settings['device'] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    return settings
