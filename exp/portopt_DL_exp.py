import torch
import torch.nn as nn

from models.portopt_DL import PortOpt_DL

class Exp_portopt_DL():
    def __init__(self, args):
        super(Exp_portopt_DL, self).__init__(args)

    def train(self, train_data, train_labels):

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

        # Create the PortOpt_DL model
        model = PortOpt_DL(input_shape, nb_stocks)

        optimizer = optim.Adam(model.parameters())

        # Training loop
        num_epochs = 100
        for epoch in range(num_epochs):
            model.train()  # Set the model to training mode
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                
               

                # Extract the last 4 features (returns)
                returns = labels[:, :, -4:]

                # Calculate portfolio returns
                portfolio_returns = torch.sum(outputs.unsqueeze(-1) * returns, dim=2)
                
                # Calculate mean and standard deviation of portfolio returns
                mean_returns = torch.mean(portfolio_returns, dim=1)
                std_returns = torch.std(portfolio_returns, dim=1)
                
                # Calculate Sharpe ratio
                sharpe_ratio = mean_returns / std_returns
                
                # Calculate the loss as the negative Sharpe ratio
                loss = -torch.mean(sharpe_ratio)

                loss.backward()
                optimizer.step()
        pass
    
    def test(self):
        pass

    def vali(self):
        pass

    def predict(self):
        pass