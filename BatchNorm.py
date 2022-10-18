import torch
from torch import nn

class BatchNorm(nn.Module):
    # `num_features`: the number of outputs for a fully connected layer
    # or the number of output channels for a convolutional layer. `num_dims`:
    # 2 for a fully connected layer and 4 for a convolutional layer
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = nn.parameter.Parameter(torch.ones(shape), requires_grad=True)
        self.beta = nn.parameter.Parameter(torch.zeros(shape), requires_grad=True)
        # The variables that are not model parameters are initialized to 0 and 1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # If `X` is not on the main memory, copy `moving_mean` and
        # `moving_var` to the device where `X` is located
        # X is a mini-batch of inputs
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # Save the updated `moving_mean` and `moving_var`
        Y, self.moving_mean, self.moving_var = self.batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.1)
        return Y

    def batch_norm(self, X, gamma, beta, moving_mean, moving_var, eps, momentum):
        if not torch.is_grad_enabled():
            # In prediction mode, use mean and variance obtained by moving average
            X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
        else:
            # In training mode
            assert len(X.shape) in (2, 4)
            if len(X.shape) == 2:
                # When using a fully connected layer, calculate the mean and
                # variance on the feature dimension
                mean = X.mean(dim=0)
                var = ((X - mean) ** 2).mean(dim=0)
            else:
                # When using a two-dimensional convolutional layer, calculate the
                # mean and variance on the channel dimension (axis=1). Here we
                # need to maintain the shape of `X`, so that the broadcasting
                # operation can be carried out later
                mean = X.mean(dim=(0, 2, 3), keepdim=True)
                var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)

            # In training mode, the current mean and variance are used
            X_hat = (X - mean) / torch.sqrt(var + eps)
            # Update the mean and variance using moving average
            moving_mean = (1.0 - momentum) * moving_mean + momentum * mean
            moving_var = (1.0 - momentum) * moving_var + momentum * var

        Y = gamma * X_hat + beta  # Scale and shift
        return Y, moving_mean.data, moving_var.data
