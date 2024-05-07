import torch
import torch.nn as nn
import torch.optim as optim

from models.portopt_DL import PortOpt_DL, PortOpt_DL_DeepSig, PortOpt_DL_DeepSig_LSTM
from torchviz import make_dot

class Exp_portopt_DL():
    def __init__(self, nb_stocks, nb_sequence):
        super(Exp_portopt_DL, self).__init__() 
        self.nb_stocks = nb_stocks
        self.nb_sequence = nb_sequence
        self.model_dict = {
            "PortOpt_DL": PortOpt_DL, 
            "PortOpt_DL_DeepSig": PortOpt_DL_DeepSig,
            "PortOpt_DL_DeepSig_LSTM": PortOpt_DL_DeepSig_LSTM
        }

    def train(self, train_loader, model_name):
        # Assuming we have a training labels of (batch x forecasting period x features)
        # We would like to extract the last 4 features which would be the return of each stock [:, :, -4:]
        # Then I would calculate the portfolio return by multiplying the weight (batch x 4) with (batch x forecast x feature)
        # in the dim = 2 so that it reduces to (32 x T x 1) -> (32 x T) tensors. Where it would be the portfolio return at each timesteps
        # after that we calculate the sharpe by mean(portfolio_returns) / std(portfolio_returns)
        # Then we backward propagate through the loss of sharpe ratio.
        
        # # Split the data into training and validation sets
        # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

        # # Create data loaders
        # train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        nb_sequence, nb_stocks = self.nb_sequence, self.nb_stocks
        Model = self.model_dict[model_name]
        # Create the PortOpt_DL model
        model = Model(nb_sequence = nb_sequence, nb_stocks = nb_stocks, in_channels = nb_stocks * 2).to('cuda:0')

        # Generate a graph of the model architecture
        optimizer = optim.Adam(model.parameters())

        # Training loop
        num_epochs = 100

        for epoch in range(num_epochs):
            print(epoch)
            model.train()  # Set the model to training mode
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs).to('cuda:0')
                # Extract the last 4 features (returns)
                returns = labels[:, :, nb_stocks:]
           
                # Calculate portfolio returns
                portfolio_returns = torch.sum(outputs.unsqueeze(1) * returns, dim = 2)
                # print(portfolio_returns)
                # Calculate mean and standard deviation of portfolio returns
                mean_returns = torch.mean(portfolio_returns, dim=1)
                std_returns = torch.std(portfolio_returns, dim=1)
                
                # Calculate Sharpe ratio
                sharpe_ratio = mean_returns / std_returns
                
                # Calculate the loss as the negative Sharpe ratio
                loss = -torch.mean(sharpe_ratio)

                loss.backward()
                optimizer.step()
        
        return model
    
    def test(self, model, test_loader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device) # Ensure the model is on the correct device

        nb_sequence, nb_stocks = self.nb_sequence, self.nb_stocks
        # Initialize variables to store the total Sharpe ratio and the number of samples
        total_sharpe_ratio = 0
        num_samples = 0

        with torch.no_grad(): # Disable gradient computation during evaluation
            for inputs, labels in test_loader:
                inputs = inputs.to(device) # Move inputs to the correct device
                labels = labels.to(device) # Move labels to the correct device

                # Pass inputs through the model
                outputs = model(inputs)
                # Assuming outputs is of shape (batch_size, 4)
                # Extract the last 4 features (returns) from labels
                returns = labels[:, :, nb_stocks:]

                # Calculate portfolio returns
                portfolio_returns = torch.sum(outputs.unsqueeze(1) * returns, dim =1)

                # Calculate mean and standard deviation of portfolio returns
                mean_returns = torch.mean(portfolio_returns, dim=1)
                std_returns = torch.std(portfolio_returns, dim=1)

                # Calculate Sharpe ratio
                sharpe_ratio = mean_returns / std_returns

                # Update total Sharpe ratio and number of samples
                total_sharpe_ratio += torch.sum(sharpe_ratio).item()
                num_samples += sharpe_ratio.numel()

            # Calculate the average Sharpe ratio
            average_sharpe_ratio = total_sharpe_ratio / num_samples
            
            # Print the average sharpe ratio
            print("Average Sharpe", average_sharpe_ratio)

            return average_sharpe_ratio
    
    def predict(self, model, test_loader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device) # Ensure the model is on the correct device

        nb_sequence, nb_stocks = self.nb_sequence, self.nb_stocks
        # Initialize variables to store the total Sharpe ratio and the number of samples
        total_sharpe_ratio = 0
        num_samples = 0

        nb_test_loader = len(test_loader)
        with torch.no_grad(): # Disable gradient computation during evaluation
            for idx, (inputs, labels) in enumerate(test_loader):
                if idx != nb_test_loader - 1:
                    continue
                inputs = inputs.to(device) # Move inputs to the correct device
                labels = labels.to(device) # Move labels to the correct device

                # Pass inputs through the model
                outputs = model(inputs)

                # Assuming outputs is of shape (batch_size, 4)
                # Extract the last 4 features (returns) from labels
                returns = labels[:, :, nb_stocks:]
                # print("outputs shape", outputs.shape)
                # print(returns.shape)
                # print("output unsqueeze", (outputs.unsqueeze(1) * returns).shape)
                # Calculate portfolio returns
                portfolio_returns = torch.sum(outputs.unsqueeze(1) * returns, dim = 2)
                # print(portfolio_returns)
                # Calculate mean and standard deviation of portfolio returns
                mean_returns = torch.mean(portfolio_returns, dim=1)
                std_returns = torch.std(portfolio_returns, dim=1)
                # print(mean_returns)
                # print(std_returns)
                # Calculate Sharpe ratio
                sharpe_ratio = mean_returns / std_returns

                # Update total Sharpe ratio and number of samples
                total_sharpe_ratio += torch.sum(sharpe_ratio).item()
                num_samples += sharpe_ratio.numel()
                break

            # Calculate the average Sharpe ratio
            average_sharpe_ratio = total_sharpe_ratio / num_samples
            
            # Print the average sharpe ratio
            print(average_sharpe_ratio)

            return outputs