import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from torch.nn.modules import loss
from torch.optim import Adam
from sklearn.preprocessing import MinMaxScaler

class Lstm_Model(nn.Module):
    def __init__(self,input_size,seq_len):
        super(Lstm_Model,self).__init__()
        self.seq_len = seq_len
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size,32,1,batch_first=True)
        self.fc = nn.Linear(32,1)
    
    def forward(self,x):
        h0 = torch.zeros((1,x.size(0),32))
        c0 = torch.zeros((1,x.size(0),32))
        out,_ = self.lstm(x,(h0,c0))
        out1 = self.fc(out)
        return out1[:,-1,]

def to_seq(dataset,seq_len=1):
    x=[]
    y=[]
    for i in range(len(dataset)-seq_len-1):
        x.append(dataset[i:seq_len+i])
        y.append(dataset[i+seq_len])
    return np.array(x),np.array(y)

df = pd.read_csv('AirPassengers.csv',usecols=[1])
plt.plot(df)
# plt.show()

dataset = df.values
scaler = MinMaxScaler(feature_range=(0,1))
dataset = dataset.astype(float)
scaler.fit_transform(dataset)
data_x,data_y = to_seq(dataset,10)
trainx,trainy = data_x[:110],data_y[110]
testx,testy = data_x[110:],data_y[110:]
model = Lstm_Model(1,10)
criterion =nn.MSELoss()
opti = Adam(model.parameters(),lr=0.001)
for epoch in range(10):
    for i in range(len(trainx)):
        opti.zero_grad()
        x,y = torch.from_numpy(trainx[i:i+1]).float(),torch.from_numpy(trainy[i:i+1]).float()
        out = model(x)
        loss = criterion(out,y)
        loss.backward()
        opti.step()
    test=[]
    
    out = model(torch.from_numpy(trainx).float())
    out= out.detach().numpy()
    test= scaler.inverse_transform(out)
    plt.plot(test)
    plt.show()
print()