import torch
import torch.nn as nn
import torch.nn.functional as F
import signatory

class PortOpt_DL(nn.Module):
    def __init__(self, nb_sequence, nb_stocks, hidden_units = 64, in_channels = 8):
        super(PortOpt_DL, self).__init__()

        self.in_channels = in_channels
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
    def __init__(self, nb_sequence, nb_stocks, sig_depth = 3, in_channels = 8):
        super(PortOpt_DL_DeepSig, self).__init__()
        
        self.batchnorm = nn.BatchNorm1d(int(nb_sequence))
        
        # To prevent having two many signatures, we try to reduce our final output as 5 nodes.
        # Kernel size is chosen to be 5 because we takes a week into consideration.
        self.augment_1 = signatory.Augment(in_channels= in_channels,
                                         layer_sizes=(10, 5),
                                         kernel_size=5,
                                         include_original=True,
                                         include_time=True)
        
        # The first lift up and transform the signatures
        self.signature_1 = signatory.Signature(depth=sig_depth, stream = True)
        
        # +6 because signatory.Augment is used to add time, and 5 other channels
        sig_channels_1 = signatory.signature_channels(channels= in_channels + 6, depth=sig_depth)

        self.augment_2 = signatory.Augment(in_channels = sig_channels_1,
                                            layer_sizes = (64, 8),
                                            kernel_size = 5,
                                            include_original = False,
                                            include_time = True)
        
        self.signature_2 = signatory.Signature(depth = sig_depth, stream = False)
        
        sig_channels_2 = signatory.signature_channels(channels= 8 + 1, depth=sig_depth)
        
        # Final Linear Layers
        self.linear = torch.nn.Linear(sig_channels_2, nb_stocks)

    def forward(self, x):

        # To turn it into float and set the device to cuda
        x = x.float().to('cuda:0')
        # Batch Normalization of the input data
        x = self.batchnorm(x)
        # Learning the features via augmentations 
        x = self.augment_1(x)

        x = self.signature_1(x)
  
        x = self.augment_2(x)
        
        x = self.signature_2(x)

        out = self.linear(x)

        out = F.softmax(out, dim=1)

        return out
    
class PortOpt_DL_DeepSig_LSTM(nn.Module):
    def __init__(self, nb_sequence, nb_stocks, sig_depth = 3, in_channels = 8):
        super(PortOpt_DL_DeepSig_LSTM, self).__init__()
        
        self.bn_0 = nn.BatchNorm1d(int(nb_sequence))
        
        # To prevent having two many signatures, we try to reduce our final output as 5 nodes.
        # Kernel size is chosen to be 5 because we takes a week into consideration.
        self.augment_1 = signatory.Augment(in_channels= in_channels,
                                         layer_sizes=(64, 8),
                                         kernel_size= 5,
                                         include_original=True,
                                         include_time=True)
        
        self.bn_1 = nn.BatchNorm1d(int(nb_sequence - 5 + 1))
        # The first lift up and transform the signatures
        self.signature_1 = signatory.Signature(depth=sig_depth, stream = True)
        
        # +6 because signatory.Augment is used to add time, and 5 other channels
        sig_channels_1 = signatory.signature_channels(channels= in_channels + 9, depth=sig_depth)

        self.lstm_1 = nn.LSTM(sig_channels_1, 20, num_layers= 2, batch_first=True)

        self.bn_2 = nn.BatchNorm1d(int(nb_sequence - 5 + 1 - 1))

        self.signature_2 = signatory.Signature(depth=sig_depth, stream = True)

        sig_channels_2 = signatory.signature_channels(channels= 20, depth = sig_depth)

        self.lstm_2 = nn.LSTM(sig_channels_2, 20, num_layers = 2, batch_first = True)

        self.bn_3 = nn.BatchNorm1d(int(nb_sequence - 5 + 1 - 2))

        self.signature_3 = signatory.Signature(depth=sig_depth, stream = True)

        sig_channels_3 = signatory.signature_channels(channels= 20, depth = sig_depth)

        self.lstm_3 = nn.LSTM(sig_channels_3, 20, num_layers = 2, batch_first = True)

        self.bn_4 = nn.BatchNorm1d(int(nb_sequence - 5 + 1 - 3))
        # Final Linear Layers
        self.linear = torch.nn.Linear(20, nb_stocks)

    def forward(self, x):
        # To turn it into float and set the device to cuda
        x = x.float().to('cuda:0')
        # Batch Normalization of the input data
        x = self.bn_0(x)
        # Learning the features via augmentations
        x = self.augment_1(x)
        # Batch Normalization after augmentations
        x = self.bn_1(x)
        # Signature computation
        x = self.signature_1(x)
        # LSTM 1
        x, _ = self.lstm_1(x)
        # Batch Normalization after LSTM 1
        x = self.bn_2(x)
        # Signature computation
        x = self.signature_2(x)
        # LSTM 2
        x, _ = self.lstm_2(x)
        # Batch Normalization after LSTM 2
        x = self.bn_3(x)
        # Signature computation
        x = self.signature_3(x)
        # LSTM 3
        x, _ = self.lstm_3(x)

        # Batch Normalization after LSTM 3
        x = self.bn_4(x)

        # Extract the last representation of LSTM
        x = x[:, -1, :] 

        # Final Linear Layers
        out = self.linear(x)

        # Softmax the output
        out = F.softmax(out, dim=1)
        
        return out