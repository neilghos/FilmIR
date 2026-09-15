"""Train a FiLM adapter on source queries and evaluate it zero-shot.

The pooled protocol uses a fixed 5,000-query subset from each source dataset:

    python train_transfer.py --source-datasets msmarco fiqa fever hotpotqa \\
        nfcorpus scifact --source-queries 5000 \\
        --target-datasets arguana scidocs

The target datasets are used only for frozen evaluation.  Their queries and
qrels never enter adapter training, negative mining, or hyperparameter
selection.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from run_beir_baselines import (
    CACHE_VERSION,
    DATASETS,
    MODELS,
    download_dataset,
    encode,
    evaluate,
    load_encoder,
    retrieve,
)
from train_film import (
    load_beir_split,
    load_or_encode_corpus,
    load_or_encode_queries,
    make_training_examples,
    retrieve_film_variants,
    train_conditioner,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/beir")
    parser.add_argument("--cache-dir", default="runs/beir_cache")
    parser.add_argument("--output-dir", default="runs/transfer_film")
    parser.add_argument(
        "--source-datasets",
        nargs="+",
        choices=list(DATASETS),
        default=None,
    )
    # Backward-compatible alias for the original one-source experiment.
    parser.add_argument("--source-dataset", choices=list(DATASETS), default=None)
    parser.add_argument(
        "--target-datasets",
        nargs="+",
        choices=list(DATASETS),
        default=["arguana", "scidocs"],
    )
    parser.add_argument("--model", choices=list(MODELS), default="minilm")
    parser.add_argument(
        "--source-queries",
        "--source-queries-per-dataset",
        dest="source_queries_per_dataset",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--source-document-pool",
        type=int,
        default=100_000,
        help="Total bounded source document pool, including positives.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hard-negatives", type=int, default=31)
    parser.add_argument("--random-negatives", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument(
        "--film-parameterization",
        choices=("rectangular", "polar"),
        default="polar",
    )
    parser.add_argument("--modulation-scale", type=float, default=0.25)
    parser.add_argument("--score-alpha", type=float, default=0.5)
    parser.add_argument("--modulation-regularization", type=float, default=0.02)
    parser.add_argument("--model-batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def read_source_queries_and_qrels(dataset_dir: Path):
    """Read train queries/qrels without loading the corpus into memory."""
    queries = {}
    with (dataset_dir / "queries.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            queries[str(row["_id"])] = row.get("text", "")

    qrels = defaultdict(dict)
    qrels_path = dataset_dir / "qrels" / "train.tsv"
    with qrels_path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if not fields or fields[0].lower() in {"query-id", "qid"}:
                continue
            if len(fields) >= 4:
                query_id, document_id, grade = fields[0], fields[2], fields[3]
            elif len(fields) == 3:
                query_id, document_id, grade = fields
            else:
                continue
            qrels[str(query_id)][str(document_id)] = float(grade)
    return queries, dict(qrels)


def select_source_queries(queries, qrels, count: int, seed: int):
    eligible = [
        query_id
        for query_id in queries
        if any(float(grade) > 0 for grade in qrels.get(query_id, {}).values())
    ]
    if count > len(eligible):
        print(
            f"requested {count} source queries, but only {len(eligible)} "
            "training queries have positives; using all available queries"
        )
        count = len(eligible)
    return sorted(random.Random(seed).sample(eligible, count))


def stream_source_documents(
    dataset_dir: Path,
    required_ids: set[str],
    pool_size: int,
    seed: int,
):
    """Collect positives and a bounded random corpus pool in one pass."""
    rng = random.Random(seed)
    positives = {}
    sampled = []
    seen_nonrequired = 0
    corpus_path = dataset_dir / "corpus.jsonl"
    with corpus_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            document_id = str(row["_id"])
            if document_id in required_ids:
                positives[document_id] = row
                continue
            seen_nonrequired += 1
            if len(sampled) < pool_size:
                sampled.append((document_id, row))
            else:
                replacement = rng.randrange(seen_nonrequired)
                if replacement < pool_size:
                    sampled[replacement] = (document_id, row)

    missing = required_ids.difference(positives)
    if missing:
        raise RuntimeError(f"{len(missing)} source qrel documents were absent from corpus")

    rows = list(positives.items())
    rows.extend(sampled)
    # Deduplicate in the unlikely event a sampled row is also required.
    by_id = {document_id: row for document_id, row in rows}
    document_ids = list(by_id)
    return document_ids, [by_id[document_id] for document_id in document_ids]


def load_or_encode_source(
    dataset_dir: Path,
    dataset_name: str,
    model_key: str,
    cache_tag: str,
    query_ids: list[str],
    queries: dict[str, str],
    document_ids: list[str],
    document_rows: list[dict],
    model,
    cache_root: Path,
    batch_size: int,
):
    cache_dir = (
        cache_root
        / "transfer"
        / CACHE_VERSION
        / dataset_name
        / model_key
        / cache_tag
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    query_path = cache_dir / "source-queries.npy"
    query_ids_path = cache_dir / "source-query-ids.json"
    corpus_path = cache_dir / "source-corpus.npy"
    corpus_ids_path = cache_dir / "source-document-ids.json"
    spec = MODELS[model_key]

    if query_path.exists() and query_ids_path.exists():
        with query_ids_path.open(encoding="utf-8") as handle:
            cached_query_ids = json.load(handle)
        if cached_query_ids != query_ids:
            raise ValueError("Cached source query selection does not match the current manifest")
        query_embeddings = np.load(query_path)
    else:
        query_embeddings = encode(
            model,
            [queries[query_id] for query_id in query_ids],
            spec["query_prefix"],
            batch_size,
            side="query",
        )
        np.save(query_path, query_embeddings.astype(np.float32))
        with query_ids_path.open("w", encoding="utf-8") as handle:
            json.dump(query_ids, handle)

    if corpus_path.exists() and corpus_ids_path.exists():
        with corpus_ids_path.open(encoding="utf-8") as handle:
            cached_document_ids = json.load(handle)
        if cached_document_ids != document_ids:
            raise ValueError("Cached source document pool does not match the current manifest")
        document_embeddings = np.load(corpus_path)
    else:
        document_embeddings = encode(
            model,
            document_rows,
            spec["document_prefix"],
            batch_size,
            side="document",
        )
        np.save(corpus_path, document_embeddings.astype(np.float32))
        with corpus_ids_path.open("w", encoding="utf-8") as handle:
            json.dump(document_ids, handle)
    return query_embeddings, document_embeddings


def main():
    args = parse_args()
    source_datasets = args.source_datasets or (
        [args.source_dataset] if args.source_dataset else
        ["msmarco", "fiqa", "fever", "hotpotqa", "nfcorpus", "scifact"]
    )
    overlap = set(source_datasets).intersection(args.target_datasets)
    if overlap:
        raise ValueError(
            "Source and zero-shot target datasets overlap: "
            + ", ".join(sorted(overlap))
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(args.data_dir)
    cache_root = Path(args.cache_dir)
    output_root = Path(args.output_dir)

    model = load_encoder(args.model, device)
    all_query_embeddings = []
    all_document_embeddings = []
    all_examples = []
    source_stats = []
    query_offset = 0
    document_offset = 0

    for source_index, source_name in enumerate(source_datasets):
        source_dir = download_dataset(source_name, data_root)
        source_queries, source_qrels = read_source_queries_and_qrels(source_dir)
        source_query_ids = select_source_queries(
            source_queries,
            source_qrels,
            args.source_queries_per_dataset,
            args.seed + source_index,
        )
        source_positive_ids = {
            document_id
            for query_id in source_query_ids
            for document_id, grade in source_qrels.get(query_id, {}).items()
            if grade > 0
        }
        source_document_ids, source_document_rows = stream_source_documents(
            source_dir,
            source_positive_ids,
            max(0, args.source_document_pool - len(source_positive_ids)),
            args.seed + source_index,
        )
        cache_tag = (
            f"q{args.source_queries_per_dataset}_pool"
            f"{args.source_document_pool}_seed{args.seed + source_index}"
        )
        source_query_embeddings, source_document_embeddings = load_or_encode_source(
            source_dir,
            source_name,
            args.model,
            cache_tag,
            source_query_ids,
            source_queries,
            source_document_ids,
            source_document_rows,
            model,
            cache_root,
            args.model_batch_size,
        )
        source_baseline_results = retrieve(
            source_query_embeddings,
            source_document_embeddings,
            source_query_ids,
            source_document_ids,
            args.top_k,
        )
        document_to_index = {
            document_id: index
            for index, document_id in enumerate(source_document_ids)
        }
        local_examples = make_training_examples(
            source_query_ids,
            source_qrels,
            source_baseline_results,
            document_to_index,
            len(source_document_ids),
            args.hard_negatives,
            args.random_negatives,
            args.seed + source_index,
        )
        all_examples.extend(
            (
                query_index + query_offset,
                [document_index + document_offset for document_index in candidates],
            )
            for query_index, candidates in local_examples
        )
        all_query_embeddings.append(source_query_embeddings)
        all_document_embeddings.append(source_document_embeddings)
        stats = {
            "dataset": source_name,
            "queries": len(source_query_ids),
            "documents": len(source_document_ids),
            "examples": len(local_examples),
        }
        source_stats.append(stats)
        print(
            f"source={source_name}/{args.model}: "
            f"queries={stats['queries']}, documents={stats['documents']}, "
            f"examples={stats['examples']}"
        )
        query_offset += len(source_query_ids)
        document_offset += len(source_document_ids)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not all_examples:
        raise RuntimeError("No source training examples were created")

    source_query_embeddings = np.concatenate(all_query_embeddings, axis=0)
    source_document_embeddings = np.concatenate(all_document_embeddings, axis=0)
    manifest = {
        "source_datasets": source_datasets,
        "source_queries_per_dataset": args.source_queries_per_dataset,
        "source_stats": source_stats,
        "source_example_count": len(all_examples),
        "seed": args.seed,
        "model": args.model,
        "target_datasets": args.target_datasets,
    }
    conditioner_args = argparse.Namespace(**vars(args))
    conditioner = train_conditioner(
        source_query_embeddings,
        source_document_embeddings,
        all_examples,
        source_query_embeddings.shape[1],
        conditioner_args,
        device,
    )

    result_root = (
        output_root
        / f"pooled_{len(source_datasets)}x{args.source_queries_per_dataset}"
        / args.model
    )
    result_root.mkdir(parents=True, exist_ok=True)
    with (result_root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    torch.save(
        {
            "model": conditioner.state_dict(),
            "embedding_dim": source_query_embeddings.shape[1],
            "model_name": MODELS[args.model]["name"],
            "source_datasets": source_datasets,
            "source_queries_per_dataset": args.source_queries_per_dataset,
            "source_document_pool": args.source_document_pool,
            "film_parameterization": args.film_parameterization,
            "modulation_scale": args.modulation_scale,
            "score_alpha": args.score_alpha,
        },
        result_root / "film.pt",
    )

    for target_name in args.target_datasets:
        target_dir = download_dataset(target_name, data_root)
        _, target_queries, target_qrels = load_beir_split(target_dir, "test")
        model = load_encoder(args.model, device)
        target_documents, target_document_ids = load_or_encode_corpus(
            target_name,
            target_dir,
            args.model,
            cache_root,
            model,
            args.model_batch_size,
        )
        target_embeddings, target_query_ids = load_or_encode_queries(
            target_name,
            target_dir,
            args.model,
            "test",
            cache_root,
            model,
            args.model_batch_size,
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        (
            baseline_results,
            film_results,
            film_only_results,
        ) = retrieve_film_variants(
            conditioner,
            target_embeddings,
            target_documents,
            target_query_ids,
            target_document_ids,
            args.top_k,
            device,
            score_alpha=args.score_alpha,
        )
        metrics = {
            "baseline": evaluate(target_qrels, baseline_results),
            "film_mixed": evaluate(target_qrels, film_results),
            "film_only": evaluate(target_qrels, film_only_results),
            "source_datasets": source_datasets,
            "source_queries_per_dataset": args.source_queries_per_dataset,
            "target_dataset": target_name,
            "model": args.model,
            "target_supervision_used": False,
        }
        target_root = result_root / target_name
        target_root.mkdir(parents=True, exist_ok=True)
        with (target_root / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        with (target_root / "baseline.run.json").open("w", encoding="utf-8") as handle:
            json.dump(baseline_results, handle)
        with (target_root / "run.json").open("w", encoding="utf-8") as handle:
            json.dump(film_results, handle)
        with (target_root / "film_only.run.json").open("w", encoding="utf-8") as handle:
            json.dump(film_only_results, handle)
        print(
            json.dumps(
                {
                    "target": target_name,
                    "baseline_ndcg@10": metrics["baseline"]["NDCG"].get("NDCG@10"),
                    "film_mixed_ndcg@10": metrics["film_mixed"]["NDCG"].get("NDCG@10"),
                    "film_only_ndcg@10": metrics["film_only"]["NDCG"].get("NDCG@10"),
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
