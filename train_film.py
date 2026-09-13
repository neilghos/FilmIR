"""Train and evaluate FiLM adapters on cached BEIR dense embeddings.

The base query/document encoder is frozen.  Only the FiLM conditioner is
trained, so any gain can be attributed to query-conditioned representation
space rather than a newly fine-tuned retriever.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from film import FiLMConditioner
from run_beir_baselines import (
    DATASETS,
    MODELS,
    download_dataset,
    encode,
    entity_text,
    evaluate,
    retrieve,
)


class CandidateDataset(Dataset):
    def __init__(self, query_indices, candidate_indices):
        self.query_indices = np.asarray(query_indices, dtype=np.int64)
        self.candidate_indices = np.asarray(candidate_indices, dtype=np.int64)

    def __len__(self):
        return len(self.query_indices)

    def __getitem__(self, index):
        return self.query_indices[index], self.candidate_indices[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/beir")
    parser.add_argument("--cache-dir", default="runs/beir_cache")
    parser.add_argument("--output-dir", default="runs/beir_film")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--train-split", default="auto", choices=("auto", "train", "dev", "test"))
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=0)
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
        help="Bound FiLM modulation directly or in polar coordinates.",
    )
    parser.add_argument(
        "--modulation-scale",
        type=float,
        default=0.25,
        help="Maximum absolute FiLM delta/beta before score mixing.",
    )
    parser.add_argument(
        "--score-alpha",
        type=float,
        default=0.5,
        help="Weight of the FiLM score correction; 0 is the baseline.",
    )
    parser.add_argument("--modulation-regularization", type=float, default=0.02)
    parser.add_argument("--model-batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def load_beir_split(dataset_dir: Path, split: str):
    from beir.datasets.data_loader import GenericDataLoader

    return GenericDataLoader(data_folder=str(dataset_dir)).load(split=split)


def load_or_encode_queries(
    dataset_name: str,
    dataset_dir: Path,
    model_key: str,
    split: str,
    cache_root: Path,
    model,
    batch_size: int,
):
    cache_dir = cache_root / dataset_name / model_key
    query_path = cache_dir / f"queries-{split}.npy"
    ids_path = cache_dir / f"query-ids-{split}.json"
    if query_path.exists() and ids_path.exists():
        with ids_path.open(encoding="utf-8") as handle:
            query_ids = json.load(handle)
        return np.load(query_path), query_ids

    # Reuse the test-query cache written by run_beir_baselines.py.  This keeps
    # the baseline and FiLM evaluation bit-for-bit aligned and avoids a second
    # encoder pass over the same queries.
    if split == "test":
        baseline_query_path = cache_dir / "queries.npy"
        baseline_ids_path = cache_dir / "ids.json"
        if baseline_query_path.exists() and baseline_ids_path.exists():
            with baseline_ids_path.open(encoding="utf-8") as handle:
                baseline_ids = json.load(handle).get("query_ids")
            if baseline_ids is not None:
                return np.load(baseline_query_path), baseline_ids

    _, queries, _ = load_beir_split(dataset_dir, split)
    query_ids = list(queries)
    spec = MODELS[model_key]
    embeddings = encode(
        model,
        [queries[query_id] for query_id in query_ids],
        spec["query_prefix"],
        batch_size,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(query_path, embeddings)
    with ids_path.open("w", encoding="utf-8") as handle:
        json.dump(query_ids, handle)
    return embeddings, query_ids


def load_or_encode_corpus(
    dataset_name: str,
    dataset_dir: Path,
    model_key: str,
    cache_root: Path,
    model,
    batch_size: int,
):
    cache_dir = cache_root / dataset_name / model_key
    embedding_path = cache_dir / "corpus.npy"
    ids_path = cache_dir / "ids.json"
    corpus, _, _ = load_beir_split(dataset_dir, "test")
    document_ids = list(corpus)
    if embedding_path.exists() and ids_path.exists():
        with ids_path.open(encoding="utf-8") as handle:
            cached_ids = json.load(handle)["document_ids"]
        if cached_ids != document_ids:
            raise ValueError(f"Cached entity ordering mismatch for {dataset_name}/{model_key}")
        return np.load(embedding_path, mmap_mode="r"), document_ids

    spec = MODELS[model_key]
    embeddings = encode(
        model,
        [entity_text(corpus[document_id]) for document_id in document_ids],
        spec["document_prefix"],
        batch_size,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(embedding_path, embeddings.astype(np.float16))
    with ids_path.open("w", encoding="utf-8") as handle:
        json.dump({"document_ids": document_ids}, handle)
    return np.load(embedding_path, mmap_mode="r"), document_ids


def choose_protocol(dataset_dir: Path, requested: str):
    if requested != "auto":
        return requested, False
    for candidate in ("train", "dev"):
        try:
            load_beir_split(dataset_dir, candidate)
            return candidate, False
        except Exception:
            pass
    return "test", True


def split_query_ids(query_ids, folds: int, fold: int, seed: int):
    if not 0 <= fold < folds:
        raise ValueError(f"--fold must be between 0 and {folds - 1}")
    order = np.random.default_rng(seed).permutation(len(query_ids))
    parts = np.array_split(order, folds)
    evaluation_indices = set(parts[fold].tolist())
    train_ids = [query_ids[i] for i in range(len(query_ids)) if i not in evaluation_indices]
    eval_ids = [query_ids[i] for i in parts[fold].tolist()]
    return train_ids, eval_ids


def make_training_examples(
    query_ids,
    qrels,
    baseline_results,
    document_to_index,
    num_documents,
    hard_negatives,
    random_negatives,
    seed,
):
    rng = random.Random(seed)
    query_indices = {query_id: i for i, query_id in enumerate(query_ids)}
    examples = []
    target_negatives = min(
        hard_negatives + random_negatives,
        max(0, num_documents - 1),
    )
    for query_id in query_ids:
        positives = [
            document_to_index[document_id]
            for document_id, grade in qrels.get(query_id, {}).items()
            if grade > 0 and document_id in document_to_index
        ]
        if not positives:
            continue
        positive_ids = set(positives)
        hard = [
            document_to_index[document_id]
            for document_id in baseline_results.get(query_id, {})
            if document_id in document_to_index
            and document_to_index[document_id] not in positive_ids
        ][:hard_negatives]
        negatives = list(dict.fromkeys(hard))[:target_negatives]
        while len(negatives) < target_negatives:
            candidate = rng.randrange(num_documents)
            if candidate not in positive_ids and candidate not in negatives:
                negatives.append(candidate)
        for positive in positives:
            examples.append((query_indices[query_id], [positive, *negatives]))
    return examples


def train_conditioner(
    query_embeddings,
    document_embeddings,
    examples,
    embedding_dim,
    args,
    device,
):
    query_tensor = torch.from_numpy(np.asarray(query_embeddings, dtype=np.float32))
    document_tensor = torch.from_numpy(np.asarray(document_embeddings, dtype=np.float32))
    dataset = CandidateDataset(
        [example[0] for example in examples],
        [example[1] for example in examples],
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    conditioner = FiLMConditioner(
        embedding_dim,
        args.hidden_dim,
        modulation_scale=args.modulation_scale,
        parameterization=args.film_parameterization,
    ).to(device)
    optimizer = torch.optim.AdamW(conditioner.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        conditioner.train()
        total_loss = 0.0
        for query_indices, candidate_indices in loader:
            queries = F.normalize(query_tensor[query_indices].to(device), dim=-1)
            candidates = document_tensor[candidate_indices].to(device)
            conditioned = conditioner.condition(queries, candidates)
            base_scores = torch.einsum("bd,bkd->bk", queries, candidates)
            film_scores = torch.einsum("bd,bkd->bk", queries, conditioned)
            scores = base_scores + args.score_alpha * (film_scores - base_scores)
            positive_scores = scores[:, :1]
            negative_scores = scores[:, 1:]
            margins = positive_scores - negative_scores
            bpr_loss = -F.logsigmoid(margins).mean()
            regularization = conditioner.regularization_loss(queries)
            loss = bpr_loss + args.modulation_regularization * regularization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        print(f"  epoch {epoch}: loss={total_loss / max(1, len(loader)):.4f}")
    return conditioner


def film_retrieve(
    conditioner,
    query_embeddings,
    document_embeddings,
    query_ids,
    document_ids,
    top_k,
    device,
    score_alpha=0.1,
):
    conditioner.eval()
    documents = torch.from_numpy(np.asarray(document_embeddings, dtype=np.float32)).to(device)
    queries = torch.from_numpy(np.asarray(query_embeddings, dtype=np.float32))
    results = {}
    with torch.no_grad():
        for query_id, query in zip(query_ids, queries):
            query = F.normalize(query[None].to(device), dim=-1)
            conditioned = conditioner.condition(query, documents[None])[0]
            base_scores = torch.sum(query * documents, dim=-1)
            film_scores = torch.sum(query * conditioned, dim=-1)
            scores = base_scores + score_alpha * (film_scores - base_scores)
            values, indices = torch.topk(scores, k=min(top_k, len(document_ids)))
            results[query_id] = {
                document_ids[int(index)]: float(value)
                for value, index in zip(values.cpu(), indices.cpu())
            }
    return results


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cache_root = Path(args.cache_dir)
    output_root = Path(args.output_dir)

    from sentence_transformers import SentenceTransformer

    for dataset_name in args.datasets:
        dataset_dir = download_dataset(dataset_name, Path(args.data_dir))
        train_split, cross_validate = choose_protocol(dataset_dir, args.train_split)
        eval_split = args.eval_split
        if cross_validate:
            _, all_queries, all_qrels = load_beir_split(dataset_dir, "test")
            train_ids, eval_ids = split_query_ids(
                list(all_queries), args.folds, args.fold, args.seed
            )
            train_qrels = all_qrels
            eval_qrels = all_qrels
        else:
            _, train_queries, train_qrels = load_beir_split(dataset_dir, train_split)
            _, eval_queries, eval_qrels = load_beir_split(dataset_dir, eval_split)
            train_ids = list(train_queries)
            eval_ids = list(eval_queries)

        for model_key in args.models:
            spec = MODELS[model_key]
            model = SentenceTransformer(spec["name"], device=device)
            documents, document_ids = load_or_encode_corpus(
                dataset_name,
                dataset_dir,
                model_key,
                cache_root,
                model,
                args.model_batch_size,
            )
            train_embeddings, cached_train_ids = load_or_encode_queries(
                dataset_name, dataset_dir, model_key, train_split, cache_root, model, args.model_batch_size
            )
            eval_embeddings, cached_eval_ids = load_or_encode_queries(
                dataset_name, dataset_dir, model_key, eval_split, cache_root, model, args.model_batch_size
            )
            train_position = {query_id: i for i, query_id in enumerate(cached_train_ids)}
            eval_position = {query_id: i for i, query_id in enumerate(cached_eval_ids)}
            train_embeddings = train_embeddings[[train_position[q] for q in train_ids]]
            eval_embeddings = eval_embeddings[[eval_position[q] for q in eval_ids]]
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            baseline_results = retrieve(
                train_embeddings,
                documents,
                train_ids,
                document_ids,
                args.top_k,
            )
            document_to_index = {document_id: i for i, document_id in enumerate(document_ids)}
            examples = make_training_examples(
                train_ids,
                train_qrels,
                baseline_results,
                document_to_index,
                len(document_ids),
                args.hard_negatives,
                args.random_negatives,
                args.seed,
            )
            if not examples:
                raise RuntimeError(f"No training examples found for {dataset_name}/{model_key}")

            print(
                f"{dataset_name}/{model_key}: train={len(train_ids)}, "
                f"eval={len(eval_ids)}, examples={len(examples)}"
            )
            conditioner = train_conditioner(
                train_embeddings,
                documents,
                examples,
                train_embeddings.shape[1],
                args,
                device,
            )
            eval_qrels_subset = {
                query_id: eval_qrels[query_id]
                for query_id in eval_ids
                if query_id in eval_qrels
            }
            baseline_eval_results = retrieve(
                eval_embeddings,
                documents,
                eval_ids,
                document_ids,
                args.top_k,
            )
            film_results = film_retrieve(
                conditioner,
                eval_embeddings,
                documents,
                eval_ids,
                document_ids,
                args.top_k,
                device,
                score_alpha=args.score_alpha,
            )
            film_only_results = film_retrieve(
                conditioner,
                eval_embeddings,
                documents,
                eval_ids,
                document_ids,
                args.top_k,
                device,
                score_alpha=1.0,
            )
            baseline_metrics = evaluate(eval_qrels_subset, baseline_eval_results)
            film_metrics = evaluate(eval_qrels_subset, film_results)
            film_only_metrics = evaluate(eval_qrels_subset, film_only_results)
            metrics = {
                "baseline": baseline_metrics,
                "film_mixed": film_metrics,
                "film_only": film_only_metrics,
                "dataset": dataset_name,
                "model": model_key,
                "train_split": train_split,
                "eval_split": eval_split,
                "cross_validation": cross_validate,
                "fold": args.fold if cross_validate else None,
                "score_alpha": args.score_alpha,
                "modulation_scale": args.modulation_scale,
                "film_parameterization": args.film_parameterization,
                "modulation_regularization": args.modulation_regularization,
            }
            result_dir = output_root / dataset_name / model_key
            result_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": conditioner.state_dict(),
                    "embedding_dim": train_embeddings.shape[1],
                    "model_name": spec["name"],
                    "train_split": train_split,
                    "fold": args.fold if cross_validate else None,
                    "modulation_scale": args.modulation_scale,
                    "film_parameterization": args.film_parameterization,
                    "score_alpha": args.score_alpha,
                },
                result_dir / "film.pt",
            )
            with (result_dir / "metrics.json").open("w", encoding="utf-8") as handle:
                json.dump(metrics, handle, indent=2)
            with (result_dir / "baseline.run.json").open("w", encoding="utf-8") as handle:
                json.dump(baseline_eval_results, handle)
            with (result_dir / "run.json").open("w", encoding="utf-8") as handle:
                json.dump(film_results, handle)
            with (result_dir / "film_only.run.json").open("w", encoding="utf-8") as handle:
                json.dump(film_only_results, handle)
            print(
                json.dumps(
                    {
                        "baseline_ndcg@10": baseline_metrics["NDCG"].get("NDCG@10"),
                        "film_mixed_ndcg@10": film_metrics["NDCG"].get("NDCG@10"),
                        "film_only_ndcg@10": film_only_metrics["NDCG"].get("NDCG@10"),
                    },
                    sort_keys=True,
                )
            )


if __name__ == "__main__":
    main()
