# -*-Encoding: utf-8 -*-
##########################################################################
#
# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
#
##########################################################################
"""
Description: Wind turbine dataset utilities
Authors: Lu,Xinjiang (luxinjiang@baidu.com)
Date:    2022/03/10
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy
import gc

# Turn off the SettingWithCopyWarning
pd.set_option('mode.chained_assignment', None)


class GlobalWindTurbineDataset(Dataset):
    """
    Desc: Data preprocessing,
          Here, e.g.    15 days for training,
                        3 days for validation,
                        and 6 days for testing
    """

    def __init__(self, data_path,
                 filename='my.csv',
                 flag='train',
                 size=None,
                 turbine_id_list=None,
                 task='MS',
                 target='Target',
                 scale=True,
                 start_col=2,  # the start column index of the data one aims to utilize
                 day_len=24 * 6,
                 train_days=15,  # 15 days
                 val_days=3,  # 3 days
                 total_days=30,  # 30 days
                 shift_days = 0
                 ):
        super().__init__()
        self.unit_size = day_len
        if size is None:
            self.input_len = self.unit_size
            self.output_len = self.unit_size
        else:
            self.input_len = size[0]
            self.output_len = size[1]
        # initialization
        assert flag in ['train', 'val']
        type_map = {'train': 0, 'val': 1}
        self.set_type = type_map[flag]
        self.task = task
        self.target = target
        self.scale = scale
        self.start_col = start_col
        self.data_path = data_path
        self.filename = filename
        self.tid_list = turbine_id_list
        self.total_size = self.unit_size * total_days
        self.train_size = train_days * self.unit_size
        self.val_size = val_days * self.unit_size
        self.shift_size = shift_days* self.unit_size

        self.__read_data__()
        self.__read_geo__()

    def __read_geo__(self):
        import json
        with open('geo_code.json','r') as f:
            geo_code=json.load(f)
        self.geo_code = geo_code['kmeans_label']
    def __read_data__(self):

        self.target_scaler_dict = {
            i: MinMaxScaler(
                clip=True) for i in self.tid_list}
        self.feature_scaler_dict = {
            i: MinMaxScaler(
                clip=True) for i in self.tid_list}
        df_raw = pd.read_csv(
            os.path.join(
                self.data_path,
                self.filename),
            on_bad_lines='skip')
        df_raw.replace(to_replace=np.nan, value=0, inplace=True)
        cols = df_raw.columns[self.start_col:]
        df_raw = df_raw[cols].astype(np.float32)
        X = deepcopy(feature_eng(df_raw.values))
        y = deepcopy(df_raw[[self.target]].values)
        del df_raw
        gc.collect()        
        self.data_x_dict = {}
        self.data_y_dict = {}
        for tid in self.tid_list:
            print(f'read data turbine {tid}...')
            border1s = [
                tid *
                self.total_size + self.shift_size,
                tid *
                self.total_size + self.shift_size +
                self.train_size -
                self.input_len]
            border2s = [tid * self.total_size +self.shift_size+ self.train_size,
                        tid * self.total_size +self.shift_size+ self.train_size + self.val_size
                        ]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            # if self.set_type==0:
            #     print(border1/self.unit_size)
            #     print(border2/self.unit_size)
            # else:
            #     print((border1+self.input_len)/self.unit_size)
            #     print(border2/self.unit_size)                
            
            # Turn off the SettingWithCopyWarning
            pd.set_option('mode.chained_assignment', None)

            train_X = X[border1s[0]:border2s[0]]
            self.feature_scaler_dict[tid].fit(train_X)
            X_scaled = self.feature_scaler_dict[tid].transform(X)
            
            train_y = y[border1s[0]:border2s[0]]
            self.target_scaler_dict[tid].fit(train_y)
            y_scaled = self.target_scaler_dict[tid].transform(y)

            data_x = X_scaled[border1:border2]
            data_y = y_scaled[border1:border2]
            self.data_x_dict[tid] = data_x
            self.data_y_dict[tid] = data_y
            del train_X, train_y, X_scaled, y_scaled,data_x,data_y
            gc.collect()
        self.feature_scaler_dict['_GLOBAL'] =  self.feature_scaler_dict[0]
        self.target_scaler_dict['_GLOBAL'] =  self.target_scaler_dict[0]
        self.size_of_both_chunks = self.input_len + 1 + self.output_len
        self.max_samples_per_ts = min(
            len(ts) for ts in self.data_y_dict.values()) - self.size_of_both_chunks
        self.ideal_nr_samples = len(self.data_y_dict) * self.max_samples_per_ts

    def __getitem__(self, index):

        # determine the index of the time series.
        tid = index // self.max_samples_per_ts
        ts_idx = index - tid * self.max_samples_per_ts
        tid_key = list(self.data_y_dict.keys())[tid]
        # # determine the index of the time series.
        s_begin = ts_idx
        s_end = s_begin + self.input_len
        r_begin = s_end
        r_end = r_begin + self.output_len
        data_x = self.data_x_dict[tid_key]
        data_y = self.data_y_dict[tid_key]
        seq_x = data_x[s_begin:s_end]
        #seq_x = np.c_[seq_x,np.ones(seq_x.shape[0])*self.geo_code[str(tid_key+1)]]  # geo code
        seq_y = data_y[r_begin:r_end]
        return seq_x, seq_y

    def __len__(self):
        return self.ideal_nr_samples

    def inverse_transform(self, data, tid):
        return self.target_scaler_dict[tid].inverse_transform(data)

def cos_transform(values,base):
    return np.cos(2*np.pi*values/base)

def feature_eng(X):
    X = pd.DataFrame(X)
    columns = ['TurbID', 'Day', 'Tmstamp', 'Wspd', 'Wdir', 'Etmp', 'Itmp', 'Ndir', 'Pab1', 'Pab2', 'Pab3', 'Prtv',
               'Patv']
    X.columns = columns[-X.shape[1]:]
    # add features
    X['Pab_avg'] = X[['Pab1', 'Pab2', 'Pab3']].mean(axis=1)
    # X['Pab_avg_cos'] = cos_transform(X['Pab_avg'],90)
    # X['Wdir_cos'] = cos_transform(X['Wdir'],360)
    # X['Ndir_cos'] = cos_transform(X['Ndir'],360)
    # # clip to normal range
    X['Wspd'] = X['Wspd'].clip(2.51,25)
    X['Wdir'] = X['Wdir'].clip(-180,180)
    X['Pab1'] = X['Pab1'].clip(0,89)
    X['Pab2'] = X['Pab2'].clip(0,89)
    X['Pab3'] = X['Pab3'].clip(0,89)
    X['Patv'] = X['Patv'].clip(10,1560)
    #X['Patv'] = inverse_sigmoid(X['Patv'])
    # select output
    out_cols = ['Wspd', 'Wdir','Patv']
    
    return X[out_cols].values




def sigmoid(x):
    x = 1/(1+np.exp(-x))
    return x

def inverse_sigmoid(x):
    x = np.log(x/(1-x))
    return x