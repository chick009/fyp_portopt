import torch 
import pandas as pd
import yfinance as yf

from data_provider.data_loader import DataPreprocessing
from data_provider.augmentation import BatchAugmentation
from exp.portopt_DL_exp import Exp_portopt_DL
from torch.utils.data import DataLoader, TensorDataset

data = DataPreprocessing('hi', True)
df = pd.read_csv('./demo_close.csv', index_col = 0)
df.index = pd.to_datetime(df.index)

# set the sequence length & forecast length & number of stocks
sequence_length = 30
forecast_length = 10
nb_degree = 0

train_data, train_label, test_data, test_label = data.run(df, 30, 30, pd.to_datetime('2020-01-01'))

device = 'cuda:0'
# Convert and move train_data, test_data, train_label, and test_label to the CUDA device
train_data, train_label, test_data, test_label = torch.from_numpy(train_data).to(device), torch.from_numpy(train_label).to(device), torch.from_numpy(test_data).to(device), torch.from_numpy(test_label).to(device)
print(test_data.shape)
# print(train_data.shape, train_label.shape)
batch_aug = BatchAugmentation()

# Append the training data in a loop
for i in range(nb_degree):
    # output = batch_aug.freq_mask(torch.tensor(train_data).to(device), torch.tensor(train_label).to(device))
    # print('output', output.shape)
    # train_append_data, train_append_label = output[:, 0:sequence_length, :].to('cuda:0'), output[:, sequence_length:, :].to('cuda:0')
    # train_data = torch.cat((torch.tensor(train_data).to('cuda:0'), train_append_data), dim=0)
    # train_label = torch.cat((torch.tensor(train_label).to('cuda:0'), train_append_data), dim=0)
    output = batch_aug.freq_mask(train_data, train_label)

    # Split the output into two parts
    train_append_data, train_append_label = output[:, 0:sequence_length, :].to('cuda:0'), output[:, sequence_length:, :].to('cuda:0')

    # Corrected concatenation for train_data
    train_data = torch.cat((train_data, train_append_data), dim=0)

    # Corrected concatenation for train_label
    # Note: This assumes train_label and train_append_label have matching sizes in all dimensions except the first one
    train_label = torch.cat((train_label, train_append_label), dim=0)

print(train_data.shape)
# Create TensorDataset for train and test data
train_dataset = TensorDataset(train_data, train_label)
test_dataset = TensorDataset(test_data, test_label)

# Create DataLoader for train and test data
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size = 1, shuffle=False)

exp = Exp_portopt_DL(8, 30)
model = exp.train(train_loader)
sharpe_ratio = exp.test(model, test_loader)