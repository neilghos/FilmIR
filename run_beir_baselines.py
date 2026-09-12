"""Evaluate pretrained dense retrieval baselines on four small BEIR datasets.

The cached embeddings are the input to the later FiLM experiment.  This
script intentionally does not train a retriever: it evaluates frozen
query/document encoders and writes their embeddings and retrieval runs.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np


# Smallest BEIR corpora by corpus size in the standard 18-dataset release.
DATASETS = {
    "nfcorpus": 3_633,
    "scifact": 5_183,
    "arguana": 8_674,
    "scidocs": 25_657,
}

MODELS = {
    "minilm": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "query_prefix": "",
        "document_prefix": "",
    },
    "multiqa": {
        "name": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        "query_prefix": "",
        "document_prefix": "",
    },
    "bge": {
        "name": "BAAI/bge-base-en-v1.5",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "document_prefix": "",
    },
    "e5": {
        "name": "intfloat/e5-base-v2",
        "query_prefix": "query: ",
        "document_prefix": "passage: ",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/beir")
    parser.add_argument("--cache-dir", default="runs/beir_cache")
    parser.add_argument("--output-dir", default="runs/beir_baselines")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-encode", action="store_true")
    return parser.parse_args()


def slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)


def download_dataset(dataset: str, root: Path, force: bool = False) -> Path:
    try:
        from beir import util
    except ImportError as exc:
        raise RuntimeError(
            "Install dependencies first: pip install -r requirements-baseline.txt"
        ) from exc

    dataset_dir = root / dataset
    if force and dataset_dir.exists():
        import shutil

        shutil.rmtree(dataset_dir)
    if not dataset_dir.exists():
        root.mkdir(parents=True, exist_ok=True)
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
        print(f"Downloading {dataset} from {url}")
        util.download_and_unzip(url, str(root))
    return dataset_dir


def entity_text(row: dict) -> str:
    title = row.get("title") or ""
    text = row.get("text") or ""
    return f"{title}. {text}".strip()


def encode(
    model,
    texts: Iterable[str],
    prefix: str,
    batch_size: int,
) -> np.ndarray:
    values = [prefix + text for text in texts]
    embeddings = model.encode(
        values,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def retrieve(
    query_embeddings: np.ndarray,
    document_embeddings: np.ndarray,
    query_ids: list[str],
    document_ids: list[str],
    top_k: int,
) -> dict[str, dict[str, float]]:
    top_k = min(top_k, len(document_ids))
    results: dict[str, dict[str, float]] = {}
    for query_id, query_embedding in zip(query_ids, query_embeddings):
        scores = np.asarray(document_embeddings, dtype=np.float32) @ query_embedding
        candidate_indices = np.argpartition(-scores, top_k - 1)[:top_k]
        candidate_indices = candidate_indices[np.argsort(-scores[candidate_indices])]
        results[query_id] = {
            document_ids[int(index)]: float(scores[int(index)])
            for index in candidate_indices
        }
    return results


def evaluate(qrels: dict, results: dict) -> dict[str, dict[str, float]]:
    try:
        from beir.retrieval.evaluation import EvaluateRetrieval
    except ImportError as exc:
        raise RuntimeError(
            "Install dependencies first: pip install -r requirements-baseline.txt"
        ) from exc
    evaluator = EvaluateRetrieval()
    ndcg, mean_average_precision, recall, precision = evaluator.evaluate(
        qrels, results, [1, 3, 5, 10, 100]
    )
    return {
        "NDCG": ndcg,
        "MAP": mean_average_precision,
        "Recall": recall,
        "P": precision,
    }


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_dir)
    cache_root = Path(args.cache_dir)
    output_root = Path(args.output_dir)

    try:
        from beir.datasets.data_loader import GenericDataLoader
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Install dependencies first: pip install -r requirements-baseline.txt"
        ) from exc

    for dataset_name in args.datasets:
        dataset_dir = download_dataset(dataset_name, data_root, args.force_download)
        corpus, queries, qrels = GenericDataLoader(
            data_folder=str(dataset_dir)
        ).load(split="test")

        document_ids = list(corpus)
        query_ids = list(queries)
        document_texts = [entity_text(corpus[doc_id]) for doc_id in document_ids]
        query_texts = [queries[query_id] for query_id in query_ids]
        print(
            f"{dataset_name}: {len(document_ids):,} documents, "
            f"{len(query_ids):,} test queries"
        )

        for model_key in args.models:
            spec = MODELS[model_key]
            model_cache = cache_root / dataset_name / model_key
            model_cache.mkdir(parents=True, exist_ok=True)
            corpus_path = model_cache / "corpus.npy"
            queries_path = model_cache / "queries.npy"
            ids_path = model_cache / "ids.json"

            if (
                corpus_path.exists()
                and queries_path.exists()
                and ids_path.exists()
                and not args.force_encode
            ):
                print(f"Loading cached embeddings: {dataset_name}/{model_key}")
                document_embeddings = np.load(corpus_path, mmap_mode="r")
                query_embeddings = np.load(queries_path)
            else:
                print(f"Encoding {dataset_name} with {spec['name']}")
                model = SentenceTransformer(spec["name"], device=args.device)
                document_embeddings = encode(
                    model,
                    document_texts,
                    spec["document_prefix"],
                    args.batch_size,
                )
                query_embeddings = encode(
                    model,
                    query_texts,
                    spec["query_prefix"],
                    args.batch_size,
                )
                np.save(corpus_path, document_embeddings.astype(np.float16))
                np.save(queries_path, query_embeddings.astype(np.float32))
                with ids_path.open("w", encoding="utf-8") as handle:
                    json.dump(
                        {"document_ids": document_ids, "query_ids": query_ids},
                        handle,
                    )
                document_embeddings = np.load(corpus_path, mmap_mode="r")

            results = retrieve(
                query_embeddings,
                document_embeddings,
                query_ids,
                document_ids,
                args.top_k,
            )
            metrics = evaluate(qrels, results)
            result_dir = output_root / dataset_name
            result_dir.mkdir(parents=True, exist_ok=True)
            with (result_dir / f"{model_key}.metrics.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(metrics, handle, indent=2)
            with (result_dir / f"{model_key}.run.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(results, handle)
            print(json.dumps(metrics["NDCG"], sort_keys=True))


if __name__ == "__main__":
    main()
