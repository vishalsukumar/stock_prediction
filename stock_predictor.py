import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from torch.nn.modules import loss
from torch.optim import Adam,LBFGS,lr_scheduler
from torch.utils.data import Dataset,DataLoader,SequentialSampler,Subset
from dataset import *
from model import Lstm_Model



data_filename = 'D:/dev/stock_prediction/aapl.us.txt'
data_set = Stockdata_set(data_filename,10)
train_set = Subset(data_set,range(0,int(len(data_set)*0.8)))
test_set = Subset(data_set,range(int(len(data_set)*0.8),len(data_set)))
train_loader = DataLoader(train_set,batch_size=256,shuffle=False)
test_loader = DataLoader(test_set,batch_size=len(test_set),shuffle=False)

model = Lstm_Model(1,32,2)
criterion =nn.MSELoss()
opti = Adam(model.parameters(),lr=0.0002)
scheduler = lr_scheduler.MultiStepLR(opti,[200],0.5)


for epoch in range(200):
    running_loss = 0
    for trainx,trainy in train_loader:
        trainx = trainx.transpose(0,1)
        opti.zero_grad()
        out = model(trainx)
        loss = torch.abs(out-trainy).sum()
        loss.backward()
        opti.step()
        running_loss+=loss.item()
    print(f'epoch = {epoch} loss= {running_loss}')
    scheduler.step()

with torch.no_grad():
    y = []
    for x_test,y_test in test_loader:
        x_test = x_test.transpose(0,1)
        y = y_test
        pred = model(x_test)
    pred = pred.detach().numpy()
    pred = pred * (data_set.max_ - data_set.min_) + data_set.min_
    y = y * (data_set.max_ - data_set.min_) + data_set.min_
    plt.plot(pred,label='Prediction')
    plt.plot(y,label='Ground truth')
    plt.legend()
    plt.show()

