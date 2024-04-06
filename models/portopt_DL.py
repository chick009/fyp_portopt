import torch
import torch.nn as nn
import torch.nn.functional as F
import signatory

class PortOpt_DL(nn.Module):
    def __init__(self, nb_sequence, nb_stocks, hidden_units = 64):
        super(PortOpt_DL, self).__init__()

        self.nb_stocks = nb_stocks 

        self.batchnorm = nn.BatchNorm1d(int(nb_sequence))
        self.lstm = nn.LSTM(nb_stocks * 2, hidden_units, batch_first=True)
        self.linear = nn.Linear(hidden_units, nb_stocks)
        
    def forward(self, x):

        x = x.float().to('cuda:0')

        # Perform batch normalization for the data
        x = self.batchnorm(x)

        # Calculate the LSTM layers
        lstm_out, _ = self.lstm(x)

        # Extract the last representation of LSTM
        last_rep = lstm_out[:, -1, :] 

        # Learn the final weight from the data 
        out = self.linear(last_rep)

        # Softmax to scale it with a total weight of 1.
        out = F.softmax(out, dim=1)

        return out

class PortOpt_DL_DeepSig(nn.Module):
    def __init__(self, input_shape, nb_stocks, in_channels, sig_depth):
        super(PortOpt_DL_DeepSig, self).__init__()
        
        self.batchnorm = nn.BatchNorm1d(30)

        self.augment = signatory.Augment(in_channels=in_channels,
                                         layer_sizes=(8, 8, 2),
                                         kernel_size=4,
                                         include_original=True,
                                         include_time=True)
        
        self.signature = signatory.Signature(depth=sig_depth)
        # +3 because signatory.Augment is used to add time, and 2 other channels
        sig_channels = signatory.signature_channels(channels=in_channels + 3, depth=sig_depth)
        self.linear = torch.nn.Linear(sig_channels, nb_stocks)

    def forward(self, x):
        x = self.augment(x)
        y = self.signature(x, basepoint = True)
        z = self.linear(y)
        return z 
    
class PortOpt_DL_DeepSig_LSTM(nn.Module):
    def __init__(self, input_shape, nb_stocks, in_channels, sig_depth):
        super(PortOpt_DL_DeepSig, self).__init__()
        
        self.batchnorm = nn.BatchNorm1d(30)

        self.augment = signatory.Augment(in_channels=in_channels,
                                         layer_sizes=(8, 8, 2),
                                         kernel_size=4,
                                         include_original=True,
                                         include_time=True)
        
        self.signature = signatory.Signature(depth=sig_depth)
        # +3 because signatory.Augment is used to add time, and 2 other channels
        sig_channels = signatory.signature_channels(channels=in_channels + 3, depth=sig_depth)
        self.linear = torch.nn.Linear(sig_channels, nb_stocks)

    def forward(self, x):
        x = self.augment(x)
        y = self.signature(x, basepoint = True)
        z = self.linear(y)
        return z 
