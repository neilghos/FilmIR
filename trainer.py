"""Train the first DBpedia-Entity Node2Vec + FiLM experiment.

This replaces the old dummy query/target tensors.  Node2Vec embeddings must
already be saved as ``Z_matrix.pt`` with rows in the same order as the BEIR
``corpus.jsonl`` entity IDs.
"""

from __future__ import annotations

import argparse
import json
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data_loader import QueryPairDataset, load_dbpedia_entity
from queryonline import FiLMQueryModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--z-matrix", default="Z_matrix.pt")
    parser.add_argument("--node-ids", default="node_ids.json")
    parser.add_argument("--mode", choices=("baseline", "film"), default="film")
    parser.add_argument("--checkpoint", default="query_model.pt")
    parser.add_argument(
        "--split",
        default="dev",
        help="BEIR split used for training; keep test held out for evaluation",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--negatives", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_dbpedia_entity(args.data_dir, split=args.split)
    node_ids_path = args.node_ids
    try:
        with open(node_ids_path, encoding="utf-8") as handle:
            node_ids = json.load(handle)
    except FileNotFoundError:
        node_ids = None
    if node_ids is not None and node_ids != data.entity_ids:
        raise ValueError(
            f"{node_ids_path} does not have the same entity ordering as corpus.jsonl"
        )
    Z = torch.load(args.z_matrix, map_location=device, weights_only=True).float()
    if Z.ndim != 2 or Z.shape[0] != data.num_entities:
        raise ValueError(
            f"Z_matrix has shape {tuple(Z.shape)} but the corpus has "
            f"{data.num_entities} entities. Node2Vec rows must follow corpus.jsonl."
        )
    Z = F.normalize(Z, dim=-1)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = FiLMQueryModel(node_dim=Z.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    dataset = QueryPairDataset(
        data,
        data.query_ids(),
        negatives_per_query=args.negatives,
        seed=args.seed,
    )

    def collate(batch: list[dict]) -> dict[str, torch.Tensor | list[str]]:
        tokens = tokenizer(
            [row["text"] for row in batch],
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        candidates = torch.tensor([row["candidates"] for row in batch], dtype=torch.long)
        return {
            "query_ids": [row["query_id"] for row in batch],
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "candidates": candidates,
            "labels": torch.zeros(len(batch), dtype=torch.long),
        }

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
    )

    print(
        f"Loaded {data.num_entities:,} entities, {len(data.queries)} queries, "
        f"device={device}"
    )
    print(f"Training Node2Vec + {args.mode} query model...")
    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            candidates = batch["candidates"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            query, gamma, _beta = model(input_ids, attention_mask)
            if args.mode == "film":
                # For dot-product retrieval, beta adds the same scalar to every
                # candidate and therefore cannot change the ranking. Gamma is
                # the meaningful first hypothesis test.
                query = query * gamma
            conditioned_query = F.normalize(query, dim=-1)
            candidate_embeddings = Z[candidates]
            scores = torch.einsum(
                "bd,bkd->bk", conditioned_query, candidate_embeddings
            )
            loss = F.cross_entropy(scores / 0.07, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch} | Loss: {total_loss / max(1, len(loader)):.4f}")

    torch.save(
        {
            "model": model.state_dict(),
            "mode": args.mode,
            "node_dim": Z.shape[1],
        },
        args.checkpoint,
    )
    print(f"Saved checkpoint to {args.checkpoint}")


if __name__ == "__main__":
    main()
