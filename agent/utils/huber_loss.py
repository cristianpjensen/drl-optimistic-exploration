import torch


def huber_loss(u, kappa=1.0):
    return torch.where(
        torch.abs(u) <= kappa,
        0.5 * torch.square(u),
        kappa * (torch.abs(u) - 0.5 * kappa),
    )
