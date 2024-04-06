import torch 
import pandas as pd
import numpy as np
import yfinance as yf

from data_provider.data_loader import DataPreprocessing
from data_provider.augmentation import BatchAugmentation
from exp.portopt_DL_exp import Exp_portopt_DL
from torch.utils.data import DataLoader, TensorDataset


df = pd.read_csv('./demo_close.csv', index_col = 0)
df.index = pd.to_datetime(df.index)

# set the sequence length & forecast length & number of stocks
sequence_length = 50
forecast_length = 62
nb_stocks = 4
nb_degree = 0

# Test start_date, and end_date
start_date = '2021-01-01'
end_date = '2021-12-31'

# Select device for training 
device = 'cuda:0'

data = DataPreprocessing('hi', add_returns=True, downsample= False)
train_data, test_data, train_label, test_label = data.run(df, sequence_length, forecast_length, pd.to_datetime(start_date), pd.to_datetime(end_date))


# Convert and move train_data, test_data, train_label, and test_label to the CUDA device
train_data, test_data, train_label, test_label = torch.from_numpy(train_data).to(device), torch.from_numpy(test_data).to(device), torch.from_numpy(train_label).to(device), torch.from_numpy(test_label).to(device)

# print(train_data.shape, train_label.shape)
batch_aug = BatchAugmentation()

# Initialize tensors to hold all train_append_data and train_append_label tensors
train_append_data_all = torch.empty(0, sequence_length, train_data.shape[2]).to(device)
train_append_label_all = torch.empty(0, forecast_length, train_label.shape[2]).to(device)

# Append the training data in a loop
for i in range(nb_degree):
    output = batch_aug.freq_mask(train_data, train_label)

    # Split the output into two parts
    train_append_data, train_append_label = output[:, 0:sequence_length, :].to(device), output[:, sequence_length:, :].to(device)
    
    # Concatenate train_append_data and train_append_label
    train_append_data_all = torch.cat((train_append_data_all, train_append_data), dim=0)
    train_append_label_all = torch.cat((train_append_label_all, train_append_label), dim=0)

if nb_degree > 0:
    # Concatenate train_append_data_all to train_data
    train_data = torch.cat((train_data, train_append_data_all), dim=0)

    # Concatenate train_append_label_all to train_label
    train_label = torch.cat((train_label, train_append_label_all), dim=0)

# Free the space
del train_append_data_all
del train_append_label_all

# Create TensorDataset for train and test data
train_dataset = TensorDataset(train_data, train_label)
test_dataset = TensorDataset(test_data, test_label)

# Create DataLoader for train and test data
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = False)
test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = False)

# sharpe_ratio_list = []

model_type = "PortOpt_DL" # "PortOpt_DL_DeepSig", # PortOpt_DL_DeepSig_LSTM

exp = Exp_portopt_DL(nb_stocks, sequence_length)
model = exp.train(train_loader, model_type)
sharpe_ratio = exp.test(model, test_loader)
# sharpe_ratio_list.append(sharpe_ratio)
