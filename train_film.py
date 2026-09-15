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
    CACHE_VERSION,
    DATASETS,
    STREAMING_DATASETS,
    MODELS,
    download_dataset,
    encode,
    encode_corpus_streaming,
    evaluate,
    load_encoder,
    load_beir_split_metadata,
    retrieve,
)

# Fixed Promptagator-style few-shot protocol.  These are deliberately
# explicit rather than inferred from files on disk: adaptation uses the BEIR
# dev split first, then train when dev is unavailable, while test-only
# datasets draw the few-shot examples from test and remove them from the
# evaluation set.
PROMPTAGATOR_ADAPTATION_SPLITS = {
    "nfcorpus": "dev",
    "scifact": "train",
    "arguana": "test",
    "scidocs": "test",
    "fiqa": "dev",
    "trec-covid": "test",
    "webis-touche2020": "test",
    "dbpedia-entity": "dev",
    "climate-fever": "test",
    "fever": "dev",
    "hotpotqa": "dev",
}


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
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--train-queries",
        type=int,
        default=0,
        help=(
            "Few-shot adaptation query count under the fixed Promptagator "
            "protocol. Test-only datasets exclude sampled test queries from "
            "evaluation. 0 uses the full adaptation split or legacy five-fold "
            "protocol."
        ),
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hard-negatives", type=int, default=31)
    parser.add_argument("--random-negatives", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--film-chunk-size", type=int, default=65_536)
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
    parser.add_argument("--model-batch-size", type=int, default=512)
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
    cache_dir = cache_root / CACHE_VERSION / dataset_name / model_key
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

    queries, _, query_ids = load_beir_split_metadata(dataset_dir, split)
    spec = MODELS[model_key]
    embeddings = encode(
        model,
        [queries[query_id] for query_id in query_ids],
        spec["query_prefix"],
        batch_size,
        side="query",
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
    cache_dir = cache_root / CACHE_VERSION / dataset_name / model_key
    embedding_path = cache_dir / "corpus.npy"
    ids_path = cache_dir / "ids.json"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if embedding_path.exists() and ids_path.exists():
        with ids_path.open(encoding="utf-8") as handle:
            cached_ids = json.load(handle)["document_ids"]
        return np.load(embedding_path, mmap_mode="r"), cached_ids

    if dataset_name in STREAMING_DATASETS:
        embeddings, document_ids = encode_corpus_streaming(
            model,
            dataset_dir / "corpus.jsonl",
            embedding_path,
            ids_path,
            MODELS[model_key]["document_prefix"],
            batch_size,
        )
        return embeddings, document_ids

    corpus, _, _ = load_beir_split(dataset_dir, "test")
    document_ids = list(corpus)

    spec = MODELS[model_key]
    embeddings = encode(
        model,
        [corpus[document_id] for document_id in document_ids],
        spec["document_prefix"],
        batch_size,
        side="document",
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Keep the frozen base representation in float32 so the baseline and
    # FiLM scores use the same numerical values as the BEIR reference path.
    np.save(embedding_path, embeddings.astype(np.float32))
    with ids_path.open("w", encoding="utf-8") as handle:
        json.dump({"document_ids": document_ids}, handle)
    return np.load(embedding_path, mmap_mode="r"), document_ids


def split_query_ids(query_ids, folds: int, fold: int, seed: int):
    if not 0 <= fold < folds:
        raise ValueError(f"--fold must be between 0 and {folds - 1}")
    order = np.random.default_rng(seed).permutation(len(query_ids))
    parts = np.array_split(order, folds)
    evaluation_indices = set(parts[fold].tolist())
    train_ids = [query_ids[i] for i in range(len(query_ids)) if i not in evaluation_indices]
    eval_ids = [query_ids[i] for i in parts[fold].tolist()]
    return train_ids, eval_ids


def sample_query_ids(query_ids, count: int, seed: int):
    """Sample a fixed few-shot adaptation set reproducibly."""
    query_ids = list(query_ids)
    if count <= 0 or count >= len(query_ids):
        return query_ids
    return random.Random(seed).sample(query_ids, count)


def assign_zero_credit(results: dict, query_ids):
    """Keep support queries in evaluation with an empty retrieval run."""
    for query_id in query_ids:
        results[query_id] = {}


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
            # Gather only this minibatch from the disk-backed corpus. Large
            # BEIR embedding matrices must not be copied into host/GPU RAM.
            candidate_array = np.asarray(
                document_embeddings[candidate_indices.numpy()], dtype=np.float32
            )
            candidates = torch.from_numpy(candidate_array).to(device)
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


def _merge_topk(
    best_values: torch.Tensor,
    best_indices: torch.Tensor,
    scores: torch.Tensor,
    offset: int,
    top_k: int,
):
    local_k = min(top_k, scores.shape[1])
    values, indices = torch.topk(scores, k=local_k, dim=1)
    indices = indices + offset
    merged_values = torch.cat((best_values, values), dim=1)
    merged_indices = torch.cat((best_indices, indices), dim=1)
    keep = torch.topk(merged_values, k=top_k, dim=1).indices
    return (
        torch.gather(merged_values, 1, keep),
        torch.gather(merged_indices, 1, keep),
    )


def _topk_results(values, indices, query_ids, document_ids):
    results = {}
    for row, query_id in enumerate(query_ids):
        order = torch.argsort(values[row], descending=True).cpu()
        row_values = values[row].cpu()
        row_indices = indices[row].cpu()
        results[query_id] = {
            document_ids[int(row_indices[index])]: float(row_values[index])
            for index in order
        }
    return results


def retrieve_film_variants(
    conditioner,
    query_embeddings,
    document_embeddings,
    query_ids,
    document_ids,
    top_k,
    device,
    score_alpha=0.1,
    chunk_size=65_536,
    query_chunk_size=64,
):
    """Exact batched full-corpus retrieval for baseline and both FiLM scores.

    The previous implementation scanned the whole corpus separately for every
    query and separately for mixed/FiLM-only scores.  This version shares each
    document chunk across a query batch and computes all score variants in one
    pass.  The top-k results are exact; only the execution order changes.
    """
    conditioner.eval()
    raw_queries = torch.from_numpy(np.asarray(query_embeddings, dtype=np.float32))
    top_k = min(top_k, len(document_ids))
    baseline_results = {}
    film_results = {}
    film_only_results = {}
    with torch.no_grad():
        for query_start in range(0, len(query_ids), query_chunk_size):
            query_end = min(query_start + query_chunk_size, len(query_ids))
            batch_ids = query_ids[query_start:query_end]
            raw_query_batch = raw_queries[query_start:query_end].to(device)
            query_batch = F.normalize(raw_query_batch, dim=-1)
            batch_size = query_end - query_start
            # Bound the [queries, documents, dimensions] FiLM tensor.  This
            # keeps BGE/E5 batches comfortably within GPU memory while still
            # reducing corpus reads by processing many queries together.
            effective_chunk_size = min(
                chunk_size,
                max(2_048, 131_072 // max(1, batch_size)),
            )
            shape = (batch_size, top_k)
            baseline_values = torch.full(shape, -torch.inf, device=device)
            baseline_indices = torch.zeros(shape, dtype=torch.long, device=device)
            film_values = torch.full(shape, -torch.inf, device=device)
            film_indices = torch.zeros(shape, dtype=torch.long, device=device)
            film_only_values = torch.full(shape, -torch.inf, device=device)
            film_only_indices = torch.zeros(shape, dtype=torch.long, device=device)
            gamma, beta = conditioner(query_batch)
            for start in range(0, len(document_ids), effective_chunk_size):
                end = min(start + effective_chunk_size, len(document_ids))
                documents = torch.from_numpy(
                    np.asarray(document_embeddings[start:end], dtype=np.float32)
                ).to(device)
                baseline_scores = raw_query_batch @ documents.T
                film_base_scores = query_batch @ documents.T
                conditioned = F.normalize(
                    gamma[:, None, :] * documents[None, :, :]
                    + beta[:, None, :],
                    dim=-1,
                )
                film_scores = torch.einsum(
                    "bd,bcd->bc", query_batch, conditioned
                )
                mixed_scores = film_base_scores + score_alpha * (
                    film_scores - film_base_scores
                )
                baseline_values, baseline_indices = _merge_topk(
                    baseline_values,
                    baseline_indices,
                    baseline_scores,
                    start,
                    top_k,
                )
                film_values, film_indices = _merge_topk(
                    film_values, film_indices, mixed_scores, start, top_k
                )
                film_only_values, film_only_indices = _merge_topk(
                    film_only_values, film_only_indices, film_scores, start, top_k
                )
            baseline_results.update(
                _topk_results(baseline_values, baseline_indices, batch_ids, document_ids)
            )
            film_results.update(
                _topk_results(film_values, film_indices, batch_ids, document_ids)
            )
            film_only_results.update(
                _topk_results(
                    film_only_values,
                    film_only_indices,
                    batch_ids,
                    document_ids,
                )
            )
    return baseline_results, film_results, film_only_results


def film_retrieve(
    conditioner,
    query_embeddings,
    document_embeddings,
    query_ids,
    document_ids,
    top_k,
    device,
    score_alpha=0.1,
    chunk_size=65_536,
):
    """Compatibility wrapper for callers that need one FiLM score variant."""
    _, film_results, film_only_results = retrieve_film_variants(
        conditioner,
        query_embeddings,
        document_embeddings,
        query_ids,
        document_ids,
        top_k,
        device,
        score_alpha=score_alpha,
        chunk_size=chunk_size,
    )
    return film_only_results if score_alpha == 1.0 else film_results


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cache_root = Path(args.cache_dir)
    output_root = Path(args.output_dir)

    for dataset_name in args.datasets:
        dataset_dir = download_dataset(dataset_name, Path(args.data_dir))
        adaptation_split = PROMPTAGATOR_ADAPTATION_SPLITS[dataset_name]
        eval_split = "test"
        cross_validate = adaptation_split == "test"
        test_support_ids = []
        if adaptation_split == "test":
            all_queries, all_qrels, all_ids = load_beir_split_metadata(dataset_dir, "test")
            if args.train_queries > 0:
                # Promptagator's test-only protocol: use a few test examples
                # for adaptation, keep all test queries in the denominator,
                # and assign the support queries zero retrieval credit below.
                train_ids = sample_query_ids(all_ids, args.train_queries, args.seed)
                test_support_ids = list(train_ids)
                eval_ids = list(all_ids)
                cross_validate = False
            else:
                # Preserve the original five-fold protocol for unrestricted
                # legacy runs on datasets without train/dev splits.
                train_ids, eval_ids = split_query_ids(
                    all_ids, args.folds, args.fold, args.seed
                )
            train_qrels = all_qrels
            eval_qrels = all_qrels
        else:
            _, train_qrels, train_ids = load_beir_split_metadata(dataset_dir, adaptation_split)
            _, eval_qrels, eval_ids = load_beir_split_metadata(dataset_dir, eval_split)
            if args.train_queries > 0 and len(train_ids) > args.train_queries:
                train_ids = sample_query_ids(train_ids, args.train_queries, args.seed)

        for model_key in args.models:
            spec = MODELS[model_key]
            model = load_encoder(model_key, device)
            documents, document_ids = load_or_encode_corpus(
                dataset_name,
                dataset_dir,
                model_key,
                cache_root,
                model,
                args.model_batch_size,
            )
            train_embeddings, cached_train_ids = load_or_encode_queries(
                dataset_name,
                dataset_dir,
                model_key,
                adaptation_split,
                cache_root,
                model,
                args.model_batch_size,
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

            # Full-corpus hard-negative mining can require billions of dot
            # products for large BEIR corpora. Use random negatives when
            # --hard-negatives is zero; otherwise retain hard-negative mining.
            if args.hard_negatives > 0:
                baseline_results = retrieve(
                    train_embeddings,
                    documents,
                    train_ids,
                    document_ids,
                    args.top_k,
                )
            else:
                baseline_results = {}
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

            # Materialize only the documents touched by the training examples.
            # This keeps large corpus memmaps out of the training loop and
            # avoids repeated random disk reads for every minibatch.
            global_candidates = np.unique(
                np.asarray(
                    [index for _, candidates in examples for index in candidates],
                    dtype=np.int64,
                )
            )
            candidate_position = {
                int(global_index): local_index
                for local_index, global_index in enumerate(global_candidates)
            }
            local_examples = [
                (
                    query_index,
                    [candidate_position[int(index)] for index in candidates],
                )
                for query_index, candidates in examples
            ]
            training_documents = np.asarray(
                documents[global_candidates], dtype=np.float32
            )

            print(
                f"{dataset_name}/{model_key}: train={len(train_ids)}, "
                f"eval={len(eval_ids)}, examples={len(examples)}, "
                f"candidate_documents={len(global_candidates)}"
            )
            conditioner = train_conditioner(
                train_embeddings,
                training_documents,
                local_examples,
                train_embeddings.shape[1],
                args,
                device,
            )
            result_dir = output_root / dataset_name / model_key
            result_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": conditioner.state_dict(),
                    "embedding_dim": train_embeddings.shape[1],
                    "model_name": spec["name"],
                    "adaptation_split": adaptation_split,
                    "train_queries": len(train_ids),
                    "fold": args.fold if cross_validate else None,
                    "modulation_scale": args.modulation_scale,
                    "film_parameterization": args.film_parameterization,
                    "score_alpha": args.score_alpha,
                },
                result_dir / "film.pt",
            )
            eval_qrels_subset = {
                query_id: eval_qrels[query_id]
                for query_id in eval_ids
                if query_id in eval_qrels
            }
            support_set = set(test_support_ids)
            scored_positions = [
                index
                for index, query_id in enumerate(eval_ids)
                if query_id not in support_set
            ]
            scored_eval_ids = [eval_ids[index] for index in scored_positions]
            scored_eval_embeddings = eval_embeddings[scored_positions]
            print("  evaluating baseline and FiLM scores over full corpus (batched)")
            (
                baseline_eval_results,
                film_results,
                film_only_results,
            ) = retrieve_film_variants(
                conditioner,
                scored_eval_embeddings,
                documents,
                scored_eval_ids,
                document_ids,
                args.top_k,
                device,
                score_alpha=args.score_alpha,
                chunk_size=args.film_chunk_size,
            )
            if test_support_ids:
                # BEIR's evaluator averages over query IDs present in the run.
                # Empty runs therefore make these support queries explicit
                # zero-score cases instead of silently dropping them.
                assign_zero_credit(baseline_eval_results, test_support_ids)
                assign_zero_credit(film_results, test_support_ids)
                assign_zero_credit(film_only_results, test_support_ids)
            baseline_metrics = evaluate(eval_qrels_subset, baseline_eval_results)
            film_metrics = evaluate(eval_qrels_subset, film_results)
            film_only_metrics = evaluate(eval_qrels_subset, film_only_results)
            metrics = {
                "baseline": baseline_metrics,
                "film_mixed": film_metrics,
                "film_only": film_only_metrics,
                "dataset": dataset_name,
                "model": model_key,
                "adaptation_split": adaptation_split,
                "train_queries": len(train_ids),
                "eval_split": eval_split,
                "cross_validation": cross_validate,
                "test_support_zero_credit": bool(test_support_ids),
                "test_support_queries": len(test_support_ids),
                "adaptation_protocol": (
                    "few_shot_test_zero_credit"
                    if test_support_ids
                    else "few_shot_split"
                    if args.train_queries > 0
                    else "full_split_or_cross_validation"
                ),
                "fold": args.fold if cross_validate else None,
                "score_alpha": args.score_alpha,
                "modulation_scale": args.modulation_scale,
                "film_parameterization": args.film_parameterization,
                "modulation_regularization": args.modulation_regularization,
            }
            result_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": conditioner.state_dict(),
                    "embedding_dim": train_embeddings.shape[1],
                    "model_name": spec["name"],
                    "adaptation_split": adaptation_split,
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
