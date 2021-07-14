import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from torch.nn.modules import loss
from torch.optim import Adam,LBFGS,lr_scheduler
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset,DataLoader,SequentialSampler,Subset


class Stockdata_set(Dataset):
    def __init__(self,filename,sequence_length):
        self.seq_len = sequence_length
        self.df = pd.read_csv(filename,usecols=[4])
        self.data_set = self.df.values
        self.max_ = np.max(self.data_set)
        self.min_ = np.min(self.data_set)
        self.data_set = (self.data_set-self.min_)/(self.max_-self.min_)
        self.x=[]
        self.y=[]
        for i in range(len(self.data_set)-self.seq_len-1):
            self.x.append(self.data_set[i:self.seq_len+i])
            self.y.append(self.data_set[i+self.seq_len])
        self.x,self.y = np.array(self.x),np.array(self.y)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx].astype(np.float32)),torch.from_numpy(self.y[idx].astype(np.float32))
