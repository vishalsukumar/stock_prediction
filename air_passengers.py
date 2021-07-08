import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from torch.nn.modules import loss
from torch.optim import Adam,LBFGS
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
# from tensorflow import keras
# from keras.models import Sequential
# from keras.layers import LSTM,Dense

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
        return torch.from_numpy(self.df.iloc[start_idx:end_idx,4:5].to_numpy()).float(),torch.from_numpy(self.df.iloc[end_idx:end_idx+1,4].to_numpy()).float()

class Lstm_Model(nn.Module):
    def __init__(self,input_size,seq_len):
        super(Lstm_Model,self).__init__()
        self.seq_len = seq_len
        self.input_size = input_size
        # self.lstm = nn.LSTM(input_size,32,1,batch_first=True)
        self.lstm_cell = nn.LSTMCell(1,32)
        self.lstm_cell_2 = nn.LSTMCell(32,32)
        self.fc = nn.Linear(32,1)
    
    def forward(self,x,future=False):
        h0 = torch.randn((x.size(1),32))
        c0 = torch.randn((x.size(1),32))
        h1 = torch.randn((x.size(1),32))
        c1 = torch.randn((x.size(1),32))
        outs = []
        for i in range(x.size(0)):
            h0,c0 = self.lstm_cell(x[i],(h0,c0))
            h1,c1 = self.lstm_cell_2(h0,(h0,c0))
            out1 = self.fc(h1)
            outs.append(out1)
        outs = torch.stack(outs,dim=0)
        h3,c3 = h0,c0
        h4,c4 = h1,c1
        x_pred = out1
        preds = []
        if future is True:
            for i in range(50):
                h3,c3 = self.lstm_cell(x_pred,(h3,c3))
                h4,c4 = self.lstm_cell_2(h3,(h3,c3))
                x_pred = self.fc(h4)
                preds.append(x_pred)
            preds = torch.stack(preds,dim=0)
            torch.cat([outs,preds],0)
        # for inp in x.split(1,dim=1):
        #     h0,c0 = self.lstm_cell(inp,(h0,c0))
        #     h1,c1 = self.lstm_cell_2(h0,(h0,c0))
        #     out1 = self.fc(h1)
        #     outs.append(out1)
        # outs = torch.cat(outs,dim=1)
        return outs[-1,:]

def to_seq(dataset,seq_len=1):
    x=[]
    y=[]
    for i in range(len(dataset)-seq_len-1):
        x.append(dataset[i:seq_len+i])
        y.append(dataset[i+seq_len])
    return np.array(x),np.array(y)


# x = np.empty((100,1000),np.float32)
# x[:] = np.array(range(1000)) + np.random.randint(-4*20,4*20,100).reshape(100,1)
# y = np.sin(x/1.0/20).astype(np.float32)
# train_x = torch.from_numpy(y[3:,:-1])
# train_y = torch.from_numpy(y[3:,1:])
# model = Lstm_Model(1,5)
# criterion = nn.MSELoss()
# optim = LBFGS(model.parameters(),lr=0.8)

# for epoch in range(1):
#     def closure():
#         optim.zero_grad()
#         out = model(train_x)
#         loss = criterion(out,train_y)
#         loss.backward()
#         print(f'loss={loss.item()}')
#         return loss
#     optim.step(closure)
# with torch.no_grad():
#     out = model(train_x[0:1])
#     plt.plot(x[3,1:],y[3,1:])
#     plt.plot(x[3,:-1],out.squeeze())
# print()


filename = 'D:/dev/stock_prediction/aapl.us.txt'
df = pd.read_csv(filename,usecols=[4])
dataset = df.values
scaler = MinMaxScaler(feature_range=(0,1))
dataset = scaler.fit_transform(dataset)
data_x,data_y = to_seq(dataset,10)
trainx,trainy = torch.from_numpy(data_x).float().transpose(0,1),torch.from_numpy(data_y).float()
model = Lstm_Model(1,10)
criterion =nn.MSELoss()
opti = Adam(model.parameters(),lr=0.01)
for epoch in range(10):
    def closure():
        opti.zero_grad()
        out = model(trainx)
        loss = criterion(out,trainy)
        loss.backward()
        print(f'loss={loss.item()}')
        return loss
    opti.step(closure)
out = model(trainx,future=True)
out= out.detach().numpy()
test= scaler.inverse_transform(out)
plt.plot(test)
plt.plot(scaler.inverse_transform(trainy))
plt.show()
# dataset = Stockdata(filename,10)
# x,y = dataset[0]
print()




df = pd.read_csv('AirPassengers.csv',usecols=[1])
# plt.plot(df)
# plt.show()

dataset = df.values
scaler = MinMaxScaler(feature_range=(0,1))
dataset = dataset.astype(float)
dataset = scaler.fit_transform(dataset)
data_x,data_y = to_seq(dataset,10)
trainx,trainy = data_x[:110],data_y[:110]
testx,testy = data_x[110:],data_y[110:]
model = Lstm_Model(1,10)
criterion =nn.MSELoss()
opti = Adam(model.parameters(),lr=0.01)
optim = LBFGS(model.parameters(),lr=0.8)
trainx,trainy = torch.from_numpy(trainx).float().transpose(0,1),torch.from_numpy(trainy).float()
for epoch in range(100):
    def closure():
        opti.zero_grad()
        out = model(trainx)
        loss = criterion(out,trainy)
        loss.backward()
        print(f'loss={loss.item()}')
        return loss
    opti.step(closure)
    # opti.zero_grad()
    # out = model(trainx)
    # loss = criterion(out,trainy)
    # loss.backward()
    # print(f'loss={loss.item()}')
    # opti.step()
out = model(trainx)
out= out.detach().numpy()
test= scaler.inverse_transform(out)
plt.plot(test)
plt.plot(scaler.inverse_transform(trainy))
plt.show()
# for epoch in range(10):
#     for i in range(len(trainx)):
#         opti.zero_grad()
#         x,y = torch.from_numpy(trainx[i:i+1]).float(),torch.from_numpy(trainy[i:i+1]).float()
#         x = x.transpose(0,1)
#         out = model(x)
#         loss = criterion(out,y)
#         loss.backward()
#         opti.step()
#     test=[]
    
    # out = model(torch.from_numpy(trainx).float())
    # out= out.detach().numpy()
    # test= scaler.inverse_transform(out)
    # plt.plot(test)
    # plt.show()
print()