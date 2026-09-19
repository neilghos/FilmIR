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
    """A learned shared query/document coordinate map followed by FiLM."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        modulation_scale: float,
        parameterization: str,
        use_corpus_centroid: bool = False,
    ) -> None:
        super().__init__()
        self.canonicalizer = LinearCanonicalizer(embedding_dim)
        self.film = FiLMConditioner(
            embedding_dim,
            hidden_dim,
            modulation_scale=modulation_scale,
            parameterization=parameterization,
            use_corpus_centroid=use_corpus_centroid,
        )

    def query_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.canonicalizer(embeddings)

    def document_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.canonicalizer(embeddings)

    def forward(
        self, query_embeddings: torch.Tensor, corpus_centroid: torch.Tensor | None = None
    ):
        return self.film(
            self.query_embeddings(query_embeddings), corpus_centroid=corpus_centroid
        )

    def condition(
        self,
        query_embeddings: torch.Tensor,
        document_embeddings: torch.Tensor,
        corpus_centroid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.film.condition(
            self.query_embeddings(query_embeddings),
            self.document_embeddings(document_embeddings),
            corpus_centroid=corpus_centroid,
        )

    def regularization_loss(
        self, query_embeddings: torch.Tensor, corpus_centroid: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.film.regularization_loss(
            self.query_embeddings(query_embeddings), corpus_centroid=corpus_centroid
        )


class FiLMConditioner(nn.Module):
    """Generate bounded query-dependent affine parameters for documents."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 256,
        modulation_scale: float = 0.1,
        parameterization: str = "rectangular",
        use_corpus_centroid: bool = False,
        modulation_mode: str = "full",
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.modulation_scale = modulation_scale
        if parameterization not in {"rectangular", "polar"}:
            raise ValueError(f"Unknown FiLM parameterization: {parameterization}")
        self.parameterization = parameterization
        self.use_corpus_centroid = use_corpus_centroid
        if modulation_mode not in {"full", "scale_only", "shift_only"}:
            raise ValueError(f"Unknown modulation_mode: {modulation_mode}")
        self.modulation_mode = modulation_mode
        input_dim = 2 * embedding_dim if use_corpus_centroid else embedding_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * embedding_dim),
        )
        # Start at the frozen baseline: gamma=1 and beta=0.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def _prepare_inputs(
        self, query_embeddings: torch.Tensor, corpus_centroid: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.use_corpus_centroid:
            if corpus_centroid is None:
                raise ValueError("corpus_centroid is required when use_corpus_centroid=True")
            if corpus_centroid.dim() == 1:
                centroid = corpus_centroid.unsqueeze(0).expand(query_embeddings.shape[0], -1)
            else:
                centroid = corpus_centroid
            return torch.cat([query_embeddings, centroid], dim=-1)
        return query_embeddings

    def forward(
        self, query_embeddings: torch.Tensor, corpus_centroid: torch.Tensor | None = None
    ):
        inputs = self._prepare_inputs(query_embeddings, corpus_centroid)
        raw_first, raw_second = self.network(inputs).chunk(2, dim=-1)
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

        if self.modulation_mode == "scale_only":
            beta = torch.zeros_like(beta)
        elif self.modulation_mode == "shift_only":
            gamma_delta = torch.zeros_like(gamma_delta)

        return 1.0 + gamma_delta, beta

    def regularization_loss(
        self, query_embeddings: torch.Tensor, corpus_centroid: torch.Tensor | None = None
    ) -> torch.Tensor:
        gamma, beta = self(query_embeddings, corpus_centroid=corpus_centroid)
        return ((gamma - 1.0).pow(2) + beta.pow(2)).mean()

    def condition(
        self,
        query_embeddings: torch.Tensor,
        document_embeddings: torch.Tensor,
        corpus_centroid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        gamma, beta = self(query_embeddings, corpus_centroid=corpus_centroid)
        conditioned = gamma[:, None, :] * document_embeddings + beta[:, None, :]
        return F.normalize(conditioned, dim=-1)


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

    def forward(
        self, query_embeddings: torch.Tensor, corpus_centroid: torch.Tensor | None = None
    ):
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

    def regularization_loss(
        self, query_embeddings: torch.Tensor, corpus_centroid: torch.Tensor | None = None
    ) -> torch.Tensor:
        gamma, beta = self(query_embeddings, corpus_centroid=corpus_centroid)
        return ((gamma - 1.0).pow(2) + beta.pow(2)).mean()

    def condition(
        self,
        query_embeddings: torch.Tensor,
        document_embeddings: torch.Tensor,
        corpus_centroid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        gamma, beta = self(query_embeddings, corpus_centroid=corpus_centroid)
        conditioned = gamma[:, None, :] * document_embeddings + beta[:, None, :]
        return F.normalize(conditioned, dim=-1)


def build_conditioner(
    embedding_dim: int,
    hidden_dim: int = 128,
    modulation_scale: float = 0.25,
    parameterization: str = "polar",
    modulator: str = "film",
    low_rank: int = 4,
    use_corpus_centroid: bool = False,
    modulation_mode: str = "full",
) -> nn.Module:
    """Factory function to build standard FiLM or LowRank FiLM conditioners."""
    if modulator == "lowrank":
        return LowRankFiLMConditioner(
            embedding_dim=embedding_dim,
            rank=low_rank,
            modulation_scale=modulation_scale,
            parameterization=parameterization,
            modulation_mode=modulation_mode,
        )
    return FiLMConditioner(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        modulation_scale=modulation_scale,
        parameterization=parameterization,
        use_corpus_centroid=use_corpus_centroid,
        modulation_mode=modulation_mode,
    )


