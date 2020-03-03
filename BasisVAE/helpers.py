import torch
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
import torch.nn.functional as F


def ELBO_collapsed_Categorical(logits_phi, alpha, K, N):
    phi = torch.softmax(logits_phi, dim=1)

    sum_alpha = alpha.sum()
    pseudocounts = phi.sum(dim=0)
    term1 = torch.lgamma(sum_alpha) - torch.lgamma(sum_alpha + N)
    term2 = (torch.lgamma(alpha + pseudocounts) - torch.lgamma(alpha)).sum()

    E_q_logq = (phi * torch.log(phi + 1e-16)).sum()

    return -term1 - term2 + E_q_logq


def KL_standard_normal(mu, sigma):
    p = Normal(torch.zeros_like(mu), torch.ones_like(mu))
    q = Normal(mu, sigma)
    return torch.sum(torch.distributions.kl_divergence(q, p))


def NB_log_prob(x, mu, theta, eps=1e-8):
    """
    Adapted from https://github.com/YosefLab/scVI/blob/master/scvi/models/log_likelihood.py
    """

    log_theta_mu_eps = torch.log(theta + mu + eps)

    res = (
        theta * (torch.log(theta + eps) - log_theta_mu_eps)
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )

    return res

def ZINB_log_prob(x, mu, theta, pi, eps=1e-8):
    """
    Adapted from https://github.com/YosefLab/scVI/blob/master/scvi/models/log_likelihood.py
    """

    softplus_pi = F.softplus(-pi)
    log_theta_eps = torch.log(theta + eps)
    log_theta_mu_eps = torch.log(theta + mu + eps)
    pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)

    case_zero = F.softplus(pi_theta_log) - softplus_pi
    mul_case_zero = torch.mul((x < eps).type(torch.float32), case_zero)

    case_non_zero = (
        -softplus_pi
        + pi_theta_log
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )
    mul_case_non_zero = torch.mul((x > eps).type(torch.float32), case_non_zero)

    res = mul_case_zero + mul_case_non_zero

    return res
