import yfinance as yf
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import MinMaxScaler


class DataPreprocessing():

    def __init__(self, type, add_returns = False):
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

    @staticmethod
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
        
        # Initialize lists to store stacked tensors
        stacked_train_data_list = []
        stacked_train_label_list = []
        stacked_test_data_list = []
        stacked_test_label_list = []

        for df in df_list:
            train_data, test_data, train_label, test_label = self.train_test_split(df, seq_len, forecast_len, split_date)

            # Stack the tensors along the 0th dimension
            stacked_train_data = np.stack(train_data, axis=0)
            stacked_train_label = np.stack(train_label, axis=0)
            stacked_test_data = np.stack(test_data, axis=0)
            stacked_test_label = np.stack(test_label, axis=0)

            # Append the stacked tensors to the respective lists
            stacked_train_data_list.append(stacked_train_data)
            stacked_train_label_list.append(stacked_train_label)
            stacked_test_data_list.append(stacked_test_data)
            stacked_test_label_list.append(stacked_test_label)

        # Concatenate the stacked tensors into a single tensor
        stacked_train_data = np.concatenate(stacked_train_data_list, axis=0)
        stacked_train_label = np.concatenate(stacked_train_label_list, axis=0)
        stacked_test_data = np.concatenate(stacked_test_data_list, axis=0)
        stacked_test_label = np.concatenate(stacked_test_label_list, axis=0)

        
        return stacked_train_data, stacked_train_label, stacked_test_data, stacked_test_label