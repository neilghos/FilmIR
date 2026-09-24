"""FiLM adapter for frozen dense-retrieval embeddings."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def canonicalize_tensor(embeddings: torch.Tensor, mode: str = "none") -> torch.Tensor:
    """Apply a shared, parameter-free coordinate canonicalization."""
    if mode == "none":
        return embeddings
    if mode == "layernorm":
        return F.layer_norm(embeddings, (embeddings.shape[-1],))
    raise ValueError(f"Unknown embedding canonicalizer: {mode}")


def canonicalize_numpy(embeddings, mode: str = "none"):
    """NumPy counterpart used by CPU hard-negative mining."""
    if mode == "none":
        return embeddings
    if mode == "layernorm":
        values = embeddings.astype("float32", copy=False)
        mean = values.mean(axis=-1, keepdims=True)
        variance = ((values - mean) ** 2).mean(axis=-1, keepdims=True)
        values = (values - mean) / np.sqrt(variance + 1e-5)
        norms = np.linalg.norm(values, axis=-1, keepdims=True)
        return values / np.maximum(norms, 1e-12)
    raise ValueError(f"Unknown embedding canonicalizer: {mode}")


class LinearCanonicalizer(nn.Module):
    """Learn an identity-initialized shared coordinate transform."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(embedding_dim, embedding_dim, bias=False)
        nn.init.eye_(self.projection.weight)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(embeddings), dim=-1)

    def regularization_loss(self) -> torch.Tensor:
        identity = torch.eye(
            self.projection.weight.shape[0],
            device=self.projection.weight.device,
            dtype=self.projection.weight.dtype,
        )
        return (self.projection.weight - identity).pow(2).mean()


class CanonicalizedFiLM(nn.Module):
    """A learned shared query/document coordinate map followed by low-rank FiLM."""

    def __init__(self, embedding_dim: int, modulation_scale: float, parameterization: str, rank: int = 4, modulation_mode: str = "full") -> None:
        super().__init__()
        self.canonicalizer = LinearCanonicalizer(embedding_dim)
        self.film = LowRankFiLMConditioner(embedding_dim=embedding_dim, rank=rank, modulation_scale=modulation_scale, parameterization=parameterization, modulation_mode=modulation_mode)

    def query_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.canonicalizer(embeddings)

    def document_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.canonicalizer(embeddings)

    def forward(self, query_embeddings: torch.Tensor):
        return self.film(self.query_embeddings(query_embeddings))

    def condition(self, query_embeddings: torch.Tensor, document_embeddings: torch.Tensor) -> torch.Tensor:
        return self.film.condition(self.query_embeddings(query_embeddings), self.document_embeddings(document_embeddings))

    def regularization_loss(self, query_embeddings: torch.Tensor) -> torch.Tensor:
        return self.film.regularization_loss(self.query_embeddings(query_embeddings))


class LowRankFiLMConditioner(nn.Module):
    """Low-rank bottleneck FiLM conditioner for parameter-efficient adaptation."""

    def __init__(
        self,
        embedding_dim: int,
        rank: int = 4,
        modulation_scale: float = 0.25,
        parameterization: str = "polar",
        modulation_mode: str = "full",
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.rank = rank
        self.modulation_scale = modulation_scale
        if parameterization not in {"rectangular", "polar"}:
            raise ValueError(f"Unknown FiLM parameterization: {parameterization}")
        self.parameterization = parameterization
        if modulation_mode not in {"full", "scale_only", "shift_only"}:
            raise ValueError(f"Unknown modulation_mode: {modulation_mode}")
        self.modulation_mode = modulation_mode
        self.down = nn.Linear(embedding_dim, rank)
        self.act = nn.GELU()
        self.up = nn.Linear(rank, 2 * embedding_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def query_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        return embeddings

    def document_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        return embeddings

    def forward(self, query_embeddings: torch.Tensor):
        h = self.act(self.down(query_embeddings))
        raw_first, raw_second = self.up(h).chunk(2, dim=-1)
        if self.parameterization == "polar":
            radius = self.modulation_scale * torch.tanh(raw_first)
            gamma_delta = radius * torch.cos(raw_second)
            beta = radius * torch.sin(raw_second)
        else:
            gamma_delta = self.modulation_scale * torch.tanh(raw_first)
            beta = self.modulation_scale * torch.tanh(raw_second)

        if self.modulation_mode == "scale_only":
            beta = torch.zeros_like(beta)
        elif self.modulation_mode == "shift_only":
            gamma_delta = torch.zeros_like(gamma_delta)

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
