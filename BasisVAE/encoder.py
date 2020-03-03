import torch
import torch.nn as nn

from torch.nn.functional import softplus

class Encoder(nn.Module):

    def __init__(self, data_dim, hidden_dim, z_dim, nonlinearity=torch.nn.ReLU):
        """
        Encoder for the VAE (neural network that maps P-dimensional data to [mu_z, sigma_z])
        :param data_dim:
        :param hidden_dim:
        :param z_dim:
        :param nonlinearity:
        """
        super().__init__()

        self.z_dim = z_dim

        # NN mapping from (Y, x) to z
        self.mapping = torch.nn.Sequential(
            torch.nn.Linear(data_dim, hidden_dim),
            nonlinearity(),
            torch.nn.Linear(hidden_dim, 2*z_dim)
        )

    def forward(self, Y):

        out = self.mapping(Y)

        mu = out[:, 0:self.z_dim]
        sigma = 1e-6 + softplus(out[:, self.z_dim:(2 * self.z_dim)])
        return mu, sigma
