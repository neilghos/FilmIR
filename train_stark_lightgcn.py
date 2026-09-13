"""Train a structural LightGCN retriever on the STaRK benchmark.

This is the first STaRK model in this repository.  It deliberately uses only
the graph structure and internal STaRK node indices:

    learnable node-ID embeddings -> LightGCN propagation -> entity vectors
    frozen text query embeddings -> projection -> query vectors

Node text/features and FiLM are intentionally left for later experiments.
The STaRK graph is transductive: all graph edges are available while learning
the query-to-entity retrieval function, while query supervision is restricted
to the official train split.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from stark_loader import _load_queries, load_stark


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalized_adjacency(edge_index: torch.LongTensor, num_nodes: int) -> torch.Tensor:
    """Build D^{-1/2}(A+I)D^{-1/2} as a sparse tensor."""
    device = edge_index.device
    loops = torch.arange(num_nodes, dtype=torch.long, device=device)
    indices = torch.cat([edge_index, torch.stack([loops, loops])], dim=1)
    values = torch.ones(indices.shape[1], dtype=torch.float32, device=device)
    degree = torch.bincount(indices[0], weights=values, minlength=num_nodes)
    inv_sqrt_degree = degree.clamp_min(1.0).pow(-0.5)
    values = inv_sqrt_degree[indices[0]] * inv_sqrt_degree[indices[1]]
    return torch.sparse_coo_tensor(
        indices,
        values,
        size=(num_nodes, num_nodes),
        device=device,
    ).coalesce()


class LightGCN(nn.Module):
    """LightGCN over the internal contiguous STaRK node IDs."""

    def __init__(self, num_nodes: int, embedding_dim: int, layers: int) -> None:
        super().__init__()
        self.node_embedding = nn.Embedding(num_nodes, embedding_dim)
        nn.init.normal_(self.node_embedding.weight, std=0.1)
        self.layers = layers

    def entity_embeddings(self, adjacency: torch.Tensor) -> torch.Tensor:
        x = self.node_embedding.weight
        outputs = [x]
        for _ in range(self.layers):
            x = torch.sparse.mm(adjacency, x)
            outputs.append(x)
        return torch.stack(outputs, dim=0).mean(dim=0)


class QueryProjector(nn.Module):
    """Map frozen sentence embeddings into the LightGCN space."""

    def __init__(self, query_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(query_dim),
            nn.Linear(query_dim, embedding_dim),
        )

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(queries), dim=-1)


def encode_queries(
    texts: list[str],
    model_name: str,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    """Encode query text once; the language encoder remains frozen."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Install sentence-transformers to encode STaRK queries."
        ) from exc

    encoder = SentenceTransformer(model_name, device=device)
    embeddings = encoder.encode(
        texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=False,
        show_progress_bar=True,
    )
    return embeddings.detach().float().cpu()


def make_positive_positions(
    answer_ids: list[list[int]],
    candidate_ids: torch.LongTensor,
    num_nodes: int,
) -> list[list[int]]:
    """Convert internal answer node IDs to columns in the candidate matrix."""
    candidate_position = torch.full((num_nodes,), -1, dtype=torch.long)
    candidate_position[candidate_ids] = torch.arange(candidate_ids.numel())
    positions: list[list[int]] = []
    for answers in answer_ids:
        row = []
        for answer_id in answers:
            position = int(candidate_position[int(answer_id)])
            if position >= 0:
                row.append(position)
        if not row:
            raise ValueError("A query has no answer in the candidate entity set")
        positions.append(row)
    return positions


def retrieval_loss(
    query_vectors: torch.Tensor,
    entity_vectors: torch.Tensor,
    positive_positions: list[list[int]],
    temperature: float,
) -> torch.Tensor:
    """Full-candidate multi-positive softmax retrieval loss."""
    scores = query_vectors @ entity_vectors.T / temperature
    positive_scores = []
    for row, positions in enumerate(positive_positions):
        positive_scores.append(torch.logsumexp(scores[row, positions], dim=0))
    numerator = torch.stack(positive_scores)
    denominator = torch.logsumexp(scores, dim=1)
    return (denominator - numerator).mean()


@torch.no_grad()
def evaluate(
    query_vectors: torch.Tensor,
    entity_vectors: torch.Tensor,
    positive_positions: list[list[int]],
    candidate_ids: torch.LongTensor,
    batch_size: int,
) -> dict[str, float]:
    """Compute STaRK's main retrieval metrics over all candidates."""
    totals = {"mrr": 0.0, "hit@1": 0.0, "hit@5": 0.0, "recall@20": 0.0}
    count = len(positive_positions)
    candidate_ids = candidate_ids.cpu()

    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        scores = query_vectors[start:stop] @ entity_vectors.T
        batch_positions = positive_positions[start:stop]

        # Exact reciprocal rank without sorting the full 129k-item list.
        for row, positions in enumerate(batch_positions):
            positive_scores = scores[row, positions]
            ranks = 1 + (scores[row].unsqueeze(0) > positive_scores.unsqueeze(1)).sum(dim=1)
            totals["mrr"] += float((1.0 / ranks.float().min()).item())

        top_k = min(20, scores.shape[1])
        top_positions = scores.topk(top_k, dim=1).indices.cpu()
        for row, positions in enumerate(batch_positions):
            positive_set = set(positions)
            ranked = top_positions[row].tolist()
            totals["hit@1"] += float(ranked[0] in positive_set)
            totals["hit@5"] += float(bool(set(ranked[:5]) & positive_set))
            totals["recall@20"] += len(set(ranked) & positive_set) / len(positive_set)

    return {name: value / count for name, value in totals.items()}


def iter_batches(num_items: int, batch_size: int, shuffle: bool) -> Iterable[torch.Tensor]:
    indices = torch.randperm(num_items) if shuffle else torch.arange(num_items)
    for start in range(0, num_items, batch_size):
        yield indices[start : start + batch_size]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("prime", "amazon", "mag"), default="prime")
    parser.add_argument(
        "--entity-update",
        choices=("batch", "epoch"),
        default="batch",
        help=(
            "Propagate the graph per query batch or once per epoch. Epoch mode "
            "is much faster for full-graph training."
        ),
    )
    parser.add_argument("--root", default="data/stark")
    parser.add_argument("--query-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--model-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--max-train-queries", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="runs/stark_lightgcn")
    args = parser.parse_args()
    if args.eval_every < 1:
        parser.error("--eval-every must be at least 1")
    set_seed(args.seed)

    device = torch.device(args.device)
    train_data = load_stark(args.dataset, split="train", root=args.root)
    qa_dataset = train_data.qa_dataset
    val_queries = _load_queries(qa_dataset, "val")
    test_queries = _load_queries(qa_dataset, "test")
    train_queries = train_data.queries

    if args.max_train_queries is not None:
        limit = min(args.max_train_queries, len(train_queries))
        train_queries = type(train_queries)(
            train_queries.split,
            train_queries.query_ids[:limit],
            train_queries.texts[:limit],
            train_queries.answer_ids[:limit],
        )

    print(
        f"{args.dataset}: train={len(train_queries)}, val={len(val_queries)}, "
        f"test={len(test_queries)}, nodes={train_data.num_nodes}, "
        f"edges={train_data.graph.num_edges}"
    )
    print("LightGCN node IDs: internal indices 0..num_nodes-1")
    print("Edge relation types are loaded but ignored by LightGCN")

    query_texts = train_queries.texts + val_queries.texts + test_queries.texts
    query_embeddings = encode_queries(
        query_texts,
        args.query_model,
        args.model_batch_size,
        str(device),
    )
    train_q = query_embeddings[: len(train_queries)]
    val_q = query_embeddings[len(train_queries) : len(train_queries) + len(val_queries)]
    test_q = query_embeddings[len(train_queries) + len(val_queries) :]

    graph = train_data.graph
    edge_index = graph.edge_index.to(device)
    graph_operator = normalized_adjacency(edge_index, train_data.num_nodes)
    candidate_ids_cpu = graph.candidate_ids.cpu()
    candidate_ids = candidate_ids_cpu.to(device)
    train_pos = make_positive_positions(train_queries.answer_ids, candidate_ids_cpu, train_data.num_nodes)
    val_pos = make_positive_positions(val_queries.answer_ids, candidate_ids_cpu, train_data.num_nodes)
    test_pos = make_positive_positions(test_queries.answer_ids, candidate_ids_cpu, train_data.num_nodes)

    graph_model = LightGCN(
        train_data.num_nodes, args.embedding_dim, args.layers
    ).to(device)
    query_model = QueryProjector(train_q.shape[1], args.embedding_dim).to(device)
    optimizer = torch.optim.AdamW(
        list(graph_model.parameters()) + list(query_model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val = -float("inf")
    best_path = output_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        graph_model.train()
        query_model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_examples = 0

        if args.entity_update == "epoch":
            # Keep one graph forward per epoch. Query losses are evaluated on
            # a detached leaf and its accumulated gradient is applied once to
            # the graph encoder after all query minibatches finish.
            entity_vectors = F.normalize(
                graph_model.entity_embeddings(graph_operator), dim=-1
            )
            entity_for_scores = entity_vectors.detach().requires_grad_(True)

        batch_count = (len(train_queries) + args.batch_size - 1) // args.batch_size
        batch_iterator = tqdm(
            iter_batches(len(train_queries), args.batch_size, shuffle=True),
            total=batch_count,
            desc=f"epoch {epoch}/{args.epochs}",
            leave=False,
        )
        for batch_number, batch_indices in enumerate(batch_iterator):
            if args.entity_update == "batch":
                entity_vectors = F.normalize(
                    graph_model.entity_embeddings(graph_operator), dim=-1
                )
                entity_for_scores = entity_vectors
                optimizer.zero_grad(set_to_none=True)
            batch_indices_cpu = batch_indices
            batch_indices = batch_indices_cpu.to(device)
            q = query_model(train_q[batch_indices_cpu].to(device))
            batch_pos = [train_pos[int(i)] for i in batch_indices_cpu]
            loss = retrieval_loss(
                q,
                entity_for_scores[candidate_ids],
                batch_pos,
                args.temperature,
            )
            if args.entity_update == "epoch":
                loss.backward()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(graph_model.parameters()) + list(query_model.parameters()), 5.0
                )
                optimizer.step()
            n = int(batch_indices.numel())
            total_loss += float(loss.item()) * n
            total_examples += n
            batch_iterator.set_postfix(loss=f"{loss.item():.4f}")

        if args.entity_update == "epoch":
            if entity_for_scores.grad is None:
                raise RuntimeError("Epoch-level entity gradient was not accumulated")
            torch.autograd.backward(entity_vectors, entity_for_scores.grad)
            torch.nn.utils.clip_grad_norm_(
                list(graph_model.parameters()) + list(query_model.parameters()), 5.0
            )
            optimizer.step()

        epoch_loss = total_loss / total_examples
        should_validate = epoch % args.eval_every == 0 or epoch == args.epochs
        if should_validate:
            graph_model.eval()
            query_model.eval()
            with torch.no_grad():
                entity_vectors = F.normalize(
                    graph_model.entity_embeddings(graph_operator), dim=-1
                )
                entity_candidates = entity_vectors[candidate_ids]
                val_vectors = query_model(val_q.to(device))
                val_metrics = evaluate(
                    val_vectors,
                    entity_candidates,
                    val_pos,
                    candidate_ids_cpu,
                    args.eval_batch_size,
                )
            print(
                f"epoch {epoch}: loss={epoch_loss:.4f} "
                f"val_mrr={val_metrics['mrr']:.4f} "
                f"val_recall@20={val_metrics['recall@20']:.4f}"
            )
            if val_metrics["mrr"] > best_val:
                best_val = val_metrics["mrr"]
                torch.save(
                    {
                        "graph_model": graph_model.state_dict(),
                        "query_model": query_model.state_dict(),
                        "args": vars(args),
                        "best_val": best_val,
                        "best_epoch": epoch,
                    },
                    best_path,
                )
        else:
            print(f"epoch {epoch}: loss={epoch_loss:.4f}")

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    graph_model.load_state_dict(checkpoint["graph_model"])
    query_model.load_state_dict(checkpoint["query_model"])
    graph_model.eval()
    query_model.eval()
    with torch.no_grad():
        entity_vectors = F.normalize(
            graph_model.entity_embeddings(graph_operator), dim=-1
        )
        entity_candidates = entity_vectors[candidate_ids]
        test_metrics = evaluate(
            query_model(test_q.to(device)),
            entity_candidates,
            test_pos,
            candidate_ids_cpu,
            args.eval_batch_size,
        )
    result = {
        "dataset": args.dataset,
        "best_val_mrr": best_val,
        "best_epoch": checkpoint.get("best_epoch"),
        "test": test_metrics,
        "checkpoint": str(best_path),
        "node_id_space": "internal_contiguous_stark_indices",
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
