import torch
from torch._C import dtype
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader, dataset
from torch.optim import Adam
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

filename = 'aapl.us.txt'
txt = pd.read_csv(filename)

class Sine_data (Dataset):
    def __init__(self,length,seq_len):
        self.length = length
        self.seq_len = seq_len
        self.x = np.arange(0.,np.pi*(length/100.+1.),np.pi/100)
        self.y = np.sin(self.x)
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx:idx+self.seq_len]).float().unsqueeze(1),torch.from_numpy(self.y[idx+self.seq_len:idx+self.seq_len+1]).float()


class Stockdata(Dataset):
    def __init__(self,filename,sequence_length):
        self.df = pd.read_csv(filename)
        self.df['diff'] = self.df['Close']-self.df['Open']
        # conditions = [
        #     (self.df['diff']<-3.0),
        #     (self.df['diff']>=-3.0) & (self.df['diff']<-1.0),
        #     (self.df['diff']>=-1.0) & (self.df['diff']<1.0),
        #     (self.df['diff']>=1.0) & (self.df['diff']<3.0),
        #     (self.df['diff']>=3.0)
        # ]
        # value = [-2,-1,0,1,2]
        conditions = [
            (self.df['diff']<=0.),
            (self.df['diff']>0.)
        ]
        value = [0,1]
        self.df['y_true'] = np.select(conditions,value)
        self.seq_len = sequence_length
    def __len__(self):
        return self.df[:6000].shape[0]-1
    def __getitem__(self, idx):
        start_idx = idx
        end_idx = idx + self.seq_len
        return torch.from_numpy(self.df.iloc[start_idx:end_idx,1:6].to_numpy()).float(),torch.from_numpy(self.df.iloc[end_idx:end_idx+1,5].to_numpy()).long().squeeze()

class LSTM(nn.Module):
    def __init__(self,input_size,seq_len,hidden_size,num_layers,num_classes):
        super(LSTM,self).__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.lstm = nn.RNN(input_size,hidden_size,1,batch_first=True)
        self.fc = nn.Linear(self.hidden_size,1)
        self.softmax = nn.Softmax(1)
    
    def forward(self,x):
        h0 = torch.zeros((1,x.size(0),self.hidden_size))
        c0 = torch.zeros((1,x.size(0),self.hidden_size))
        out,_ = self.lstm(x,h0)
        out = out[:,-1,:]
        out1 = self.fc(out)
        return out1

def stock_loss(pred,target):
    return torch.abs(torch.exp(torch.abs(pred))-torch.exp(torch.abs(target))).sum()

# dataset = Stockdata(filename,30)
dataset = Sine_data(1000,1)
train_loader = DataLoader(dataset,32,False)
loss_fn = nn.MSELoss()
lstm = LSTM(1,1,52,1,5)
optimizer = Adam(lstm.parameters(),lr=0.0001)
for epochs in range(100):
    running_loss=0
    step=0
    for samplex,sampley in train_loader:
        optimizer.zero_grad()
        out = lstm(samplex)
        loss = loss_fn(out,sampley)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        step+=1
    running_loss /= step
    print(f'loss={running_loss}')
    outs = []
    trus = []
    for i in range(len(dataset)):
        x,y = dataset[i]
        out_ = lstm(x.unsqueeze(0))
        outs.append(out_)
        trus.append(y)
    plt.plot(range(len(outs)),outs)
    plt.show()
print()

# samplex_test,sampley_test = dataset[8000]
# samplex_test = samplex_test.unsqueeze(0)
# out_ = lstm(samplex_test)
print()
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