import torch 
import pandas as pd
import yfinance as yf

from fyp_portopt.data_provider.data_loader import DataPreprocessing
from fyp_portopt.data_provider.augmentation import BatchAugmentation

data = DataPreprocessing('hi', True)
df = pd.read_csv('./fyp_portopt/demo_close.csv', index_col = 0)
df.index = pd.to_datetime(df.index)
a, b, c, d = data.run(df, 30, 10 , pd.to_datetime('2020-01-01'))

batch_aug = BatchAugmentation()

device = 'cuda:0'
output = batch_aug.freq_mask(torch.tensor(a).to(device), torch.tensor(b).to(device))