"""Project-local loader for the STaRK KG retrieval benchmark.

This module is intentionally separate from the BEIR/IR loaders.  It exposes
the objects needed by the KG model:

    query text + answer entity IDs
    node IDs, node types, and node text access
    typed graph edges
    candidate-entity masks

The official ``stark_qa`` package performs downloading and preprocessing.  We
only normalize its objects into a stable interface for the graph-text model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import torch


STARK_DATASETS = ("amazon", "mag", "prime")
STARK_SPLITS = ("train", "val", "test", "test-0.1", "human_generated_eval")


def _require_stark():
    try:
        from stark_qa import load_qa, load_skb
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError(
            "STaRK support is optional. Install it with "
            "`python -m pip install -r requirements-stark.txt`."
        ) from exc
    return load_qa, load_skb


@dataclass
class StarkQueries:
    """A split of STaRK natural-language queries and entity answers."""

    split: str
    query_ids: list[int]
    texts: list[str]
    answer_ids: list[list[int]]

    def __len__(self) -> int:
        return len(self.query_ids)


@dataclass
class StarkGraph:
    """Normalized STaRK graph and retrieval metadata.

    Node indices are the official STaRK internal IDs.  ``edge_index`` follows
    the PyG convention with shape ``[2, num_edges]``.  STaRK's processed SKB
    is undirected by default, so ``edge_types`` is aligned with the returned
    (possibly symmetrized) edge index.
    """

    dataset: str
    num_nodes: int
    edge_index: torch.LongTensor
    edge_types: torch.LongTensor
    node_types: torch.LongTensor
    candidate_ids: torch.LongTensor
    node_type_names: dict[int, str]
    edge_type_names: dict[int, str]
    _skb: object

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def num_candidates(self) -> int:
        return int(self.candidate_ids.numel())

    @property
    def candidate_mask(self) -> torch.BoolTensor:
        mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        mask[self.candidate_ids] = True
        return mask

    def node_text(
        self,
        node_ids: Iterable[int] | None = None,
        *,
        add_rel: bool = False,
        compact: bool = True,
    ) -> list[str]:
        """Materialize text for selected nodes through the official SKB API.

        Text is deliberately lazy because Amazon and MAG contain millions of
        nodes.  Call this on candidate IDs or manageable batches when creating
        text features for the encoder.
        """
        ids = self.candidate_ids.tolist() if node_ids is None else list(node_ids)
        return [
            self._skb.get_doc_info(int(node_id), add_rel=add_rel, compact=compact)
            for node_id in ids
        ]

    def node_type(self, node_id: int) -> str:
        return self.node_type_names[int(self.node_types[int(node_id)].item())]

    def edge_type(self, edge_id: int) -> str:
        return self.edge_type_names[int(self.edge_types[int(edge_id)].item())]


@dataclass
class StarkData:
    """Complete input bundle for the KG retrieval experiments."""

    dataset: str
    graph: StarkGraph
    queries: StarkQueries
    qa_dataset: object

    @property
    def num_nodes(self) -> int:
        return self.graph.num_nodes

    @property
    def num_candidates(self) -> int:
        return self.graph.num_candidates


def _load_queries(qa_dataset, split: str) -> StarkQueries:
    if split not in STARK_SPLITS:
        raise ValueError(f"Unknown STaRK split {split!r}; choose from {STARK_SPLITS}")

    split_indices = qa_dataset.get_idx_split()[split].tolist()
    query_ids: list[int] = []
    texts: list[str] = []
    answer_ids: list[list[int]] = []
    for dataset_index in split_indices:
        query, query_id, answers, _ = qa_dataset[int(dataset_index)]
        query_ids.append(int(query_id))
        texts.append(str(query))
        answer_ids.append([int(answer_id) for answer_id in answers])
    return StarkQueries(split, query_ids, texts, answer_ids)


def _load_graph(dataset: str, skb) -> StarkGraph:
    edge_index = torch.as_tensor(skb.edge_index, dtype=torch.long).contiguous()
    edge_types = torch.as_tensor(skb.edge_types, dtype=torch.long).contiguous()
    node_types = torch.as_tensor(skb.node_types, dtype=torch.long).contiguous()
    candidate_ids = torch.as_tensor(skb.get_candidate_ids(), dtype=torch.long)

    node_type_names = {
        int(node_type_id): str(node_type_name)
        for node_type_id, node_type_name in skb.node_type_dict.items()
    }
    edge_type_names = {
        int(edge_type_id): str(edge_type_name)
        for edge_type_id, edge_type_name in skb.edge_type_dict.items()
    }
    return StarkGraph(
        dataset=dataset,
        num_nodes=int(skb.num_nodes()),
        edge_index=edge_index,
        edge_types=edge_types,
        node_types=node_types,
        candidate_ids=candidate_ids,
        node_type_names=node_type_names,
        edge_type_names=edge_type_names,
        _skb=skb,
    )


def load_stark(
    dataset: Literal["amazon", "mag", "prime"],
    *,
    split: Literal["train", "val", "test", "test-0.1", "human_generated_eval"] = "train",
    root: str | Path = "data/stark",
    download_processed: bool = True,
    human_generated_eval: bool = False,
    **skb_kwargs,
) -> StarkData:
    """Load one STaRK graph and one official query split.

    ``root`` is separate from ``data/beir`` by design.  The first call may
    download the official STaRK QA files and processed SKB.  Set
    ``human_generated_eval=True`` only for the dedicated human-query split.
    """
    if dataset not in STARK_DATASETS:
        raise ValueError(f"Unknown STaRK dataset {dataset!r}; choose from {STARK_DATASETS}")
    if human_generated_eval and split != "human_generated_eval":
        raise ValueError("human_generated_eval=True requires split='human_generated_eval'")
    if split == "human_generated_eval":
        human_generated_eval = True

    load_qa, load_skb = _require_stark()
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    qa_dataset = load_qa(
        dataset,
        root=str(root / "qa"),
        human_generated_eval=human_generated_eval,
    )
    skb = load_skb(
        dataset,
        root=str(root / "skb"),
        download_processed=download_processed,
        **skb_kwargs,
    )
    return StarkData(
        dataset=dataset,
        graph=_load_graph(dataset, skb),
        queries=_load_queries(qa_dataset, split),
        qa_dataset=qa_dataset,
    )


def summarize(data: StarkData) -> dict[str, int | str]:
    """Return a compact JSON-friendly loader summary."""
    return {
        "dataset": data.dataset,
        "split": data.queries.split,
        "queries": len(data.queries),
        "queries_with_answers": sum(bool(ids) for ids in data.queries.answer_ids),
        "nodes": data.graph.num_nodes,
        "candidate_nodes": data.graph.num_candidates,
        "edges": data.graph.num_edges,
        "node_types": len(data.graph.node_type_names),
        "edge_types": len(data.graph.edge_type_names),
    }

