import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from torch.nn.modules import loss
from torch.optim import Adam,LBFGS
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM,Dense



def to_seq(dataset,seq_len=1):
    x=[]
    y=[]
    for i in range(len(dataset)-seq_len-1):
        x.append(dataset[i:seq_len+i,0])
        y.append(dataset[i+seq_len,:])
    return np.array(x),np.array(y)

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
trainx = np.reshape(trainx,(trainx.shape[0],1,trainx.shape[1]))
y = np.expand_dims(trainy,1)
keras_model = Sequential()
keras_model.add(LSTM(64,return_sequences=True,input_shape=(None,10)))
keras_model.add(LSTM(32,input_shape=(None,10)))
keras_model.add(Dense(32))
keras_model.add(Dense(1))
keras_model.compile(loss='mean_squared_error',optimizer='adam')
keras_model.fit(trainx,trainy,verbose=2,epochs=100)
out = keras_model.predict(trainx)
outs = scaler.inverse_transform(out)
plt.plot(outs)
plt.plot(scaler.inverse_transform(trainy))
plt.show()