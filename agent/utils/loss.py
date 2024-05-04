import torch


def huber_loss(u: torch.Tensor, kappa=1.0) -> torch.Tensor:
    """Computes the element-wise Huber loss."""

    return torch.where(
        torch.abs(u) <= kappa,
        0.5 * torch.square(u),
        kappa * (torch.abs(u) - 0.5 * kappa),
    )


def quantile_huber_loss(td_error_BNM: torch.Tensor, tau_BN: torch.Tensor, kappa=1.0) -> torch.Tensor:
    """Computes the quantile loss.

    Ref: https://arxiv.org/pdf/1806.06923 (Equation 3)

    Dimension keys:
        B: batch size
        N: number of quantiles
        M: number of target quantiles

    Output: [B,]
    """

    huber_BNM = huber_loss(td_error_BNM, kappa)
    quantile_BNM = torch.abs(tau_BN.unsqueeze(2) - (td_error_BNM < 0).float()) * huber_BNM / kappa

    return quantile_BNM.mean(dim=2).sum(dim=1)
