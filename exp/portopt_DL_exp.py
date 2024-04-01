import torch
import torch.nn as nn
import torch.optim as optim

from models.portopt_DL import PortOpt_DL

class Exp_portopt_DL():
    def __init__(self, nb_stocks, input_shape):
        super(Exp_portopt_DL, self).__init__() 
        self.nb_stocks = nb_stocks
        self.input_shape = input_shape

    def train(self, train_loader):
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

        input_shape, nb_stocks = self.input_shape, self.nb_stocks
        # Create the PortOpt_DL model
        model = PortOpt_DL(input_shape, nb_stocks).to('cuda:0')

        optimizer = optim.Adam(model.parameters())

        # Training loop
        num_epochs = 100

        for epoch in range(num_epochs):

            model.train()  # Set the model to training mode
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs).to('cuda:0')

                # Extract the last 4 features (returns)
                returns = labels[:, :, 4:]
           
                # Calculate portfolio returns
                portfolio_returns = torch.sum(outputs.unsqueeze(1) * returns, dim=1)
                
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
                returns = labels[:, :, 4:]

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
            print(average_sharpe_ratio)

            return average_sharpe_ratio
    def predict(self):
        pass