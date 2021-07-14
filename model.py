import numpy as np
import torch
import torch.nn as nn

class Lstm_Model(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers):
        super(Lstm_Model,self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers)
        self.fc = nn.Linear(self.hidden_size,1)
    
    def forward(self,x):
        h0 = torch.randn((self.num_layers,x.size(1),self.hidden_size))
        c0 = torch.randn((self.num_layers,x.size(1),self.hidden_size))
        out,(h0,c0) = self.lstm(x,(h0,c0))
        return self.fc(out)[-1,:]