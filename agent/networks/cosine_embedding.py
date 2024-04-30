import torch
import torch.nn as nn


class CosineEmbedding(nn.Module):
    """Compute the embedding of a sample point tau.

    Ref: https://arxiv.org/pdf/1806.06923 (Equation 4)

    Dimension keys:
        B: batch size
        N: number of taus
        E: embedding dimensionality
        D: feature dimensionality

    Output: B x N x D
    
    """

    def __init__(self, emb_dim=64, feat_dim=3136):
        super(CosineEmbedding, self).__init__()

        self.emb_indices_E = nn.Parameter(
            torch.arange(emb_dim).reshape(1, 1, -1).float(),
            requires_grad=False,
        )

        self.net = nn.Sequential(
            nn.Linear(emb_dim, feat_dim, bias=True),
            nn.ReLU(),
        )

    def forward(self, tau_BN: torch.Tensor):
        # Cosine embedding for each tau, which is passed to the network
        cos_embedding_BNE = torch.cos(torch.pi * tau_BN.unsqueeze(-1) * self.emb_indices_E)
        return self.net(cos_embedding_BNE)
