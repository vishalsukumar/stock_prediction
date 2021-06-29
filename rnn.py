import torch
from torch._C import dtype
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import numpy as np

filename = '//nas-fa0efsusr1/sukumavl/Desktop/aapl.us.txt'
txt = pd.read_csv(filename)

class Stockdata(Dataset):
    def __init__(self,filename,sequence_length):
        self.df = pd.read_csv(filename)
        self.df['diff'] = self.df['Close']-self.df['Open']
        conditions = [
            (self.df['diff']<-3.0),
            (self.df['diff']>=-3.0) & (self.df['diff']<-1.0),
            (self.df['diff']>=-1.0) & (self.df['diff']<1.0),
            (self.df['diff']>=1.0) & (self.df['diff']<3.0),
            (self.df['diff']>=3.0)
        ]
        value = [-2,-1,0,1,2]
        self.df['y_true'] = np.select(conditions,value)
        self.seq_len = sequence_length
    def __len__(self):
        return self.df.shape[0]
    def __getitem__(self, idx):
        start_idx = idx
        end_idx = idx + self.seq_len
        return torch.from_numpy(self.df.iloc[start_idx:end_idx,1:6].to_numpy()).float(),torch.from_numpy(self.df.iloc[start_idx:end_idx,8:].to_numpy()).float()

class LSTM(nn.Module):
    def __init__(self,input_size,seq_len,hidden_size,num_layers,num_classes):
        super(LSTM,self).__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.lstm = nn.RNN(5,256,1,batch_first=True)
        self.fc = nn.Linear(self.hidden_size,self.num_classes)
    
    def forward(self,x):
        h0 = torch.zeros((1,x.size(0),self.hidden_size))
        c0 = torch.zeros((1,x.size(0),self.hidden_size))
        out,_ = self.lstm(x,h0)
        out = out[:,-1,:]
        out1 = self.fc(out)
        return out1

dataset = Stockdata(filename,30) 
train_loader = DataLoader(dataset,8,True)
for samplex,sampley in train_loader:
    lstm = LSTM(5,30,256,1,5)
    out = lstm(samplex)


# txt['diff'] = txt['Close']-txt['Open']
# txt_m3_to_below = txt[(txt['diff']<-3.0)]
# txt_m3_to_m1 = txt[(txt['diff']>=-3.0) & (txt['diff']<-1.0)]
# txt_m1_to_1 = txt[(txt['diff']>=-1.0) & (txt['diff']<1.0)]
# txt_1_to_3 = txt[(txt['diff']>=1.0) & (txt['diff']<3.0)]
# txt_3_to_above = txt[(txt['diff']>=3.0)]
# print()

# conditions = [
#     (txt['diff']<-3.0),
#     (txt['diff']>=-3.0) & (txt['diff']<-1.0),
#     (txt['diff']>=-1.0) & (txt['diff']<1.0),
#     (txt['diff']>=1.0) & (txt['diff']<3.0),
#     (txt['diff']>=3.0)
# ]

# value = [-2,-1,0,1,2]

# txt['y_true'] = np.select(conditions,value)
print()