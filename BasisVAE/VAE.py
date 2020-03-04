import torch
import torch.nn as nn
import torch

from .helpers import KL_standard_normal

class VAE(nn.Module):
    """
    VAE wrapper class (for combining a standard encoder and BasisVAE decoder)
    """

    def __init__(self, encoder, decoder, lr):
        super().__init__()

        self.encoder = encoder

        self.decoder = decoder

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, Y, batch_scale=1.0, beta=1.0):
        """
        :param Y: data matrix
        :param batch_scale: scaling constant for log-likelihood (typically total_sample_size / batch_size)
        :param beta: additional constant for KL scaling (beta * KL_terms)
        :return: (mu_z, sigma_z, loss)
        """

        # encode
        mu_z, sigma_z = self.encoder(Y)
        eps = torch.randn_like(mu_z)
        z = mu_z + sigma_z * eps

        # decode
        y_pred, dropout_prob_logit, theta = self.decoder.forward(z)
        decoder_loss = self.decoder.loss(Y, y_pred, dropout_prob_logit, theta, batch_scale, beta)

        # latent space loss
        VAE_KL_loss = batch_scale * KL_standard_normal(mu_z, sigma_z)

        total_loss = decoder_loss + beta * VAE_KL_loss

        return mu_z, sigma_z, total_loss

    # map (Y, x) to z and calculate p(y* | x, z_mu)
    def calculate_test_loglik(self, Y):

        with torch.no_grad():
            mu_z, sigma_z = self.encoder(Y)
            Y_pred, dropout_prob_logit, theta = self.decoder.forward(mu_z)
            loglik = self.decoder.loglik(Y, Y_pred, dropout_prob_logit, theta)
            return loglik

    def optimize(self, data_loader, n_epochs, beta=1.0, logging_freq=10, verbose=True):

        # sample size
        N = len(data_loader.dataset)

        # scaling for loglikelihood terms
        batch_size = data_loader.batch_size
        batch_scale = N / batch_size

        if verbose:
            print(f"Fitting BasisVAE.\n"
                  f"\tScale-invariance={self.decoder.scale_invariance}"
                  f"\tTranslation-invariance={self.decoder.translation_invariance}\n")
            print(f"\tData set size {N}, batch size {batch_size}, number of basis funs {self.decoder.n_basis}.\n")

        for epoch in range(n_epochs):

            train_loss = 0.0

            for batch_idx, (Y_subset, ) in enumerate(data_loader):

                mu_z, sigma_z, loss = self.forward(Y_subset, batch_scale=batch_scale, beta=beta)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            if epoch % logging_freq == 0:
                print(f"\tEpoch: {epoch:2}. Total loss: {train_loss:11.2f}")
