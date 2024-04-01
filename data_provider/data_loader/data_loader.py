import yfinance as yf
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import MinMaxScaler

def download_stock_data(tickers, start_date, end_date):
    
    data = yf.download(tickers, start=start_date, end=end_date, interval='1d')['Adj Close']

    df = pd.DataFrame(data)
    
    return df.resample('D').last()



class DataPreprocessing():

    def __init__(self, type, add_returns):
        self.type = type 
        self.add_returns = True

    def append_return_columns(self, df):
        # Get the return dataframe by performing percentage change.
        returns_df = df.pct_change()

        # Rename the columns to indicate that they represent returns
        returns_df.columns = [col + '_return' for col in returns_df.columns]

        # Append the returns columns to the original dataframe
        df = pd.concat([df, returns_df], axis=1)

        # Drop the columns for the first one
        df = df.dropna()

        return df
    
    def downsample_list(self, df, interval_lst = ['1D', '3D', '1W']):
    
        df_list = []
    
        for interval in interval_lst: 
            
            data = df.resample(interval).last()

            if self.add_returns == True:
                data = self.append_return_columns(data)

            df_list.append(data)
        
        return df_list


    def train_test_split(df, seq_len, forecast_len, split_date):
        train_data, test_data, train_label, test_label = [], [], [], []
        
        # Convert split_date to a datetime object
        split_date = pd.Timestamp(split_date, tz='UTC')
        
        for i in range(len(df) - seq_len - forecast_len - 1):
            
            # Get the timestamp for the start of forecast
            time = df.iloc[i + seq_len].name
            
            # Compare the timestamp with the split_date
            if time < split_date:
                # Append data and label to train sets
                train_data.append(df.iloc[i: i + seq_len].values)
                train_label.append(df.iloc[i + seq_len: i + seq_len + forecast_len].values)
            else:
                # Append data and label to test sets
                test_data.append(df.iloc[i: i + seq_len].values)
                test_label.append(df.iloc[i + seq_len: i + seq_len + forecast_len].values)
        
        # Convert lists to numpy arrays
        train_data = np.array(train_data)
        test_data = np.array(test_data)
        train_label = np.array(train_label)
        test_label = np.array(test_label)
        
        return train_data, test_data, train_label, test_label
    
    def run(self, df, seq_len, forecast_len, split_date):
        """
        Stack arrays from multiple dataframes obtained from train_test_split function.
        
        Args:
            df_list (list): List of dataframes.
            seq_len (int): Length of the sequence.
            forecast_len (int): Length of the forecast.
            split_date (str): Date to split train and test sets.
        
        Returns:
            tuple: A tuple containing the stacked arrays (train_data, train_label, test_data, test_label).
        """

        # Downsample the dataset
        df_list = self.downsample_list(df, interval_lst = ['1D', '3D', '1W'])
        
        # Choose the correct conversion function
        stacked_train_data_list = []
        stacked_train_label_list = []
        stacked_test_data_list = []
        stacked_test_label_list = []
        
        for df in df_list:
            train_data, test_data, train_label, test_label = self.train_test_split(df, seq_len, forecast_len, split_date)
            
            stacked_train_data_list.append(np.vstack(train_data))
            stacked_train_label_list.append(np.vstack(train_label))
            stacked_test_data_list.append(np.vstack(test_data))
            stacked_test_label_list.append(np.vstack(test_label))
        
        stacked_train_data = np.vstack(stacked_train_data_list)
        stacked_train_label = np.vstack(stacked_train_label_list)
        stacked_test_data = np.vstack(stacked_test_data_list)
        stacked_test_label = np.vstack(stacked_test_label_list)
        
        return stacked_train_data, stacked_train_label, stacked_test_data, stacked_test_label