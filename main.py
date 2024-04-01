import pandas as pd
import yfinance as yf

def download_stock_data(tickers, start_date, end_date):

    data = yf.download(tickers, start=start_date, end=end_date, interval='1d')['Adj Close']

    df = pd.DataFrame(data)

    return df.resample('D').last().dropna()

tickers = ['VTI', 'AGG', 'DBC', '^VIX']
start_date = '2010-01-01'
end_date = '2020-12-31'

df = download_stock_data(tickers, start_date, end_date)
print(df)


import argparse
import os
import torch