"""Evaluate pretrained dense retrieval baselines on configured BEIR datasets.

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


# Bump this whenever the text serialization or encoder pooling changes.  It
# prevents stale embeddings from silently contaminating comparisons.
CACHE_VERSION = "v5_beir_reference"


# BEIR corpora currently used by the baseline and FiLM pipelines.
DATASETS = {
    "nfcorpus": 3_633,
    "scifact": 5_183,
    "arguana": 8_674,
    "scidocs": 25_657,
    "fiqa": 57_638,
    "fever": 5_416_568,
    "msmarco": 8_841_823,
    "hotpotqa": 5_233_329,
}

MODELS = {
    "minilm": {
        "kind": "sentence_transformer",
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "query_prefix": "",
        "document_prefix": "",
    },
    "multiqa": {
        "kind": "sentence_transformer",
        "name": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        "query_prefix": "",
        "document_prefix": "",
    },
    "bge": {
        "kind": "sentence_transformer",
        "name": "BAAI/bge-base-en-v1.5",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "document_prefix": "",
    },
    "e5": {
        "kind": "sentence_transformer",
        "name": "intfloat/e5-base-v2",
        "query_prefix": "query: ",
        "document_prefix": "passage: ",
    },
    "contriever": {
        "kind": "hf_shared",
        "name": "facebook/contriever",
        "query_prefix": "",
        "document_prefix": "",
    },
    "dpr": {
        "kind": "dpr",
        # This is the multi-dataset DPR pair used by the original BEIR
        # reference evaluator, not the weaker NQ-only single-nq pair.
        "name": "facebook/dpr-question_encoder-multiset-base",
        "document_name": "facebook/dpr-ctx_encoder-multiset-base",
        "query_prefix": "",
        "document_prefix": "",
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
    # Match BEIR's corpus serialization: title, one space, then text.
    return f"{title} {text}".strip()


def encode(
    model,
    texts: Iterable[str],
    prefix: str,
    batch_size: int,
    side: str = "document",
) -> np.ndarray:
    items = list(texts)
    if side == "document" and items and isinstance(items[0], dict):
        if hasattr(model, "encode_corpus"):
            embeddings = model.encode_corpus(
                items,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
            )
            if hasattr(embeddings, "detach"):
                embeddings = embeddings.detach().cpu().numpy()
            return np.asarray(embeddings, dtype=np.float32)
        values = [prefix + entity_text(item) for item in items]
    else:
        values = [prefix + text for text in items]
    if hasattr(model, "encode_side"):
        return model.encode_side(values, side=side, batch_size=batch_size)
    if side == "query" and hasattr(model, "encode_queries"):
        embeddings = model.encode_queries(
            values,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        if hasattr(embeddings, "detach"):
            embeddings = embeddings.detach().cpu().numpy()
        return np.asarray(embeddings, dtype=np.float32)
    if side == "document" and hasattr(model, "encode_corpus"):
        embeddings = model.encode_corpus(
            values,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        if hasattr(embeddings, "detach"):
            embeddings = embeddings.detach().cpu().numpy()
        return np.asarray(embeddings, dtype=np.float32)
    embeddings = model.encode(
        values,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


class HFTextEncoder:
    """Small adapter for Hugging Face encoders not packaged as SentenceTransformers."""

    def __init__(self, model_name: str, device: str, model_kind: str, document_name: str | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.device = torch.device(device)
        self.model_kind = model_kind
        self.models = {}
        self.tokenizers = {}
        if model_kind == "contriever":
            self.models["query"] = AutoModel.from_pretrained(model_name).to(self.device).eval()
            self.tokenizers["query"] = AutoTokenizer.from_pretrained(model_name)
            self.models["document"] = self.models["query"]
            self.tokenizers["document"] = self.tokenizers["query"]
        elif model_kind == "dpr":
            from transformers import (
                DPRContextEncoder,
                DPRContextEncoderTokenizerFast,
                DPRQuestionEncoder,
                DPRQuestionEncoderTokenizerFast,
            )

            if document_name is None:
                raise ValueError("DPR requires a context encoder checkpoint")
            self.models["query"] = DPRQuestionEncoder.from_pretrained(model_name).to(self.device).eval()
            self.tokenizers["query"] = DPRQuestionEncoderTokenizerFast.from_pretrained(model_name)
            self.models["document"] = DPRContextEncoder.from_pretrained(document_name).to(self.device).eval()
            self.tokenizers["document"] = DPRContextEncoderTokenizerFast.from_pretrained(document_name)
        else:
            raise ValueError(f"Unsupported Hugging Face encoder kind: {model_kind}")

    def encode_side(self, texts: list[str], side: str, batch_size: int) -> np.ndarray:
        import torch

        if side not in self.models:
            raise ValueError(f"Unsupported encoding side: {side}")
        model = self.models[side]
        tokenizer = self.tokenizers[side]
        batches = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                tokens = tokenizer(
                    texts[start : start + batch_size],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                tokens = {key: value.to(self.device) for key, value in tokens.items()}
                outputs = model(**tokens)
                # Contriever is trained/evaluated with mean pooling over the
                # masked token states.  AutoModel may expose a pooler_output
                # for its BERT backbone, but using that CLS vector collapses
                # Contriever's BEIR performance.  DPR, in contrast, defines
                # its representation as pooler_output.
                if self.model_kind == "contriever":
                    hidden = outputs.last_hidden_state
                    mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    embedding = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
                    embedding = torch.nn.functional.normalize(embedding, dim=-1)
                else:
                    embedding = outputs.pooler_output
                batches.append(embedding.cpu().numpy().astype(np.float32))
        if not batches:
            return np.empty((0, model.config.hidden_size), dtype=np.float32)
        return np.concatenate(batches, axis=0)

    def encode_corpus(self, corpus: list[dict], batch_size: int, **kwargs) -> np.ndarray:
        """Encode DPR documents using the official title/text tokenizer pair."""
        if self.model_kind != "dpr":
            return self.encode_side([entity_text(row) for row in corpus], "document", batch_size)

        import torch

        model = self.models["document"]
        tokenizer = self.tokenizers["document"]
        titles = [row.get("title") or "" for row in corpus]
        texts = [row.get("text") or "" for row in corpus]
        batches = []
        with torch.no_grad():
            for start in range(0, len(corpus), batch_size):
                tokens = tokenizer(
                    titles[start : start + batch_size],
                    texts[start : start + batch_size],
                    padding=True,
                    truncation="longest_first",
                    max_length=512,
                    return_tensors="pt",
                )
                tokens = {key: value.to(self.device) for key, value in tokens.items()}
                # Match BEIR's historical DPR wrapper exactly: it forwards
                # only input_ids and attention_mask, omitting token_type_ids.
                embedding = model(
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens["attention_mask"],
                ).pooler_output
                batches.append(embedding.cpu().numpy().astype(np.float32))
        if not batches:
            return np.empty((0, model.config.hidden_size), dtype=np.float32)
        return np.concatenate(batches, axis=0)


def load_encoder(model_key: str, device: str | None = None):
    spec = MODELS[model_key]
    kind = spec["kind"]
    if kind == "sentence_transformer":
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(spec["name"], device=device)
    if kind == "hf_shared":
        # Use BEIR's own Hugging Face wrapper for Contriever.  It implements
        # the reference mean-pooling and normalization path.
        from beir.retrieval.models import HuggingFace

        return HuggingFace(
            model_path=spec["name"],
            sep=" ",
            pooling="mean",
            # The original Contriever BEIR script defaults to raw dot-product
            # embeddings; normalization is opt-in there.
            normalize=False,
            max_length=512,
            prompts={"query": "", "passage": ""},
        )
    resolved_device = device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
    if kind == "dpr":
        return HFTextEncoder(
            spec["name"],
            resolved_device,
            "dpr",
            document_name=spec["document_name"],
        )
    raise ValueError(f"Unknown model kind: {kind}")


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
        document_rows = [corpus[doc_id] for doc_id in document_ids]
        query_texts = [queries[query_id] for query_id in query_ids]
        print(
            f"{dataset_name}: {len(document_ids):,} documents, "
            f"{len(query_ids):,} test queries"
        )

        for model_key in args.models:
            spec = MODELS[model_key]
            model_cache = cache_root / CACHE_VERSION / dataset_name / model_key
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
                model = load_encoder(model_key, args.device)
                document_embeddings = encode(
                    model,
                    document_rows,
                    spec["document_prefix"],
                    args.batch_size,
                    side="document",
                )
                query_embeddings = encode(
                    model,
                    query_texts,
                    spec["query_prefix"],
                    args.batch_size,
                    side="query",
                )
                # Keep reference embeddings in float32.  Float16 is useful for
                # large experiments but can change close retrieval ties.
                np.save(corpus_path, document_embeddings.astype(np.float32))
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
