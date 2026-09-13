"""FiLM adapter for frozen dense-retrieval embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMConditioner(nn.Module):
    """Generate bounded query-dependent affine parameters for documents."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 256,
        modulation_scale: float = 0.1,
        parameterization: str = "rectangular",
    ) -> None:
        super().__init__()
        self.modulation_scale = modulation_scale
        if parameterization not in {"rectangular", "polar"}:
            raise ValueError(f"Unknown FiLM parameterization: {parameterization}")
        self.parameterization = parameterization
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * embedding_dim),
        )
        # Start at the frozen baseline: gamma=1 and beta=0.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, query_embeddings: torch.Tensor):
        raw_first, raw_second = self.network(query_embeddings).chunk(2, dim=-1)
        if self.parameterization == "polar":
            # The modulation vector (gamma - 1, beta) lies inside a disk.
            # Zero-initialized outputs give radius=0, hence the exact baseline.
            radius = self.modulation_scale * torch.tanh(raw_first)
            gamma_delta = radius * torch.cos(raw_second)
            beta = radius * torch.sin(raw_second)
        else:
            # Keep the learned space close to the frozen baseline.  The tanh
            # bound prevents a query from arbitrarily changing dimensions.
            gamma_delta = self.modulation_scale * torch.tanh(raw_first)
            beta = self.modulation_scale * torch.tanh(raw_second)
        return 1.0 + gamma_delta, beta

    def regularization_loss(self, query_embeddings: torch.Tensor) -> torch.Tensor:
        gamma, beta = self(query_embeddings)
        return ((gamma - 1.0).pow(2) + beta.pow(2)).mean()

    def condition(
        self,
        query_embeddings: torch.Tensor,
        document_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        gamma, beta = self(query_embeddings)
        conditioned = gamma[:, None, :] * document_embeddings + beta[:, None, :]
        return F.normalize(conditioned, dim=-1)
