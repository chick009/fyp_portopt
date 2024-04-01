import torch
import torch.nn as nn

class PortOpt_DL(nn.Module):
    def __init__(self, input_shape, nb_stocks):
        super(PortOpt_DL, self).__init__()
        
        self.batchnorm = nn.BatchNorm1d(30)
        self.lstm = nn.LSTM(8, 64, batch_first=True)
        self.linear = nn.Linear(64, 4)
        
    def forward(self, x):
        x = x.float().to('cuda:0')
        x = self.batchnorm(x)
        lstm_out, _ = self.lstm(x)
        last_rep = lstm_out[:, -1, :]  # Extract the last representations
        out = self.linear(last_rep)
        return out