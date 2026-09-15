"""Evaluate pretrained dense retrieval baselines on configured BEIR datasets.

The cached embeddings are the input to the later FiLM experiment.  This
script intentionally does not train a retriever: it evaluates frozen
query/document encoders and writes their embeddings and retrieval runs.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import re
from pathlib import Path
from typing import Iterable, Iterator

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
    "trec-covid": 171_332,
    "webis-touche2020": 382_545,
    "dbpedia-entity": 4_635_922,
    "climate-fever": 5_416_593,
    "fever": 5_416_568,
    "hotpotqa": 5_233_329,
}

STREAMING_DATASETS = {
    "trec-covid",
    "webis-touche2020",
    "dbpedia-entity",
    "climate-fever",
    "fever",
    "hotpotqa",
}

DOWNLOAD_ARCHIVES = {
    "trec-covid": "trec-covid-beir",
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
    parser.add_argument(
        "--corpus-chunk-size",
        type=int,
        default=65536,
        help="Streaming corpus rows per encoding chunk for large datasets.",
    )
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

    archive = DOWNLOAD_ARCHIVES.get(dataset, dataset)
    dataset_dir = root / dataset
    archive_dir = root / archive
    if force:
        import shutil

        for candidate in {dataset_dir, archive_dir}:
            if candidate.exists() and candidate != root:
                shutil.rmtree(candidate)
    if dataset_dir.exists():
        return dataset_dir
    # Some official archives, notably TREC-COVID, extract into a directory
    # named after the archive rather than the BEIR dataset key.
    if archive_dir.exists():
        return archive_dir
    if not dataset_dir.exists():
        root.mkdir(parents=True, exist_ok=True)
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{archive}.zip"
        print(f"Downloading {dataset} from {url}")
        util.download_and_unzip(url, str(root))
    if dataset_dir.exists():
        return dataset_dir
    if archive_dir.exists():
        return archive_dir
    raise FileNotFoundError(
        f"Downloaded {dataset}, but neither {dataset_dir} nor {archive_dir} exists"
    )


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


def load_beir_split_metadata(dataset_dir: Path, split: str):
    """Load BEIR queries/qrels without materializing the corpus."""
    query_file = dataset_dir / "queries.jsonl"
    qrels_file = dataset_dir / "qrels" / f"{split}.tsv"
    if not query_file.exists() or not qrels_file.exists():
        raise FileNotFoundError(f"Missing BEIR {split} metadata in {dataset_dir}")

    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with qrels_file.open("r", encoding="utf-8") as handle:
        next(handle, None)
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) == 3:
                query_id, document_id, score = fields
            elif len(fields) == 4:
                query_id, _, document_id, score = fields
            else:
                raise ValueError(f"Unexpected qrels row in {qrels_file}: {line!r}")
            qrels[query_id][document_id] = int(score)

    queries: dict[str, str] = {}
    with query_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            query_id = str(row["_id"])
            if query_id in qrels:
                queries[query_id] = row.get("text") or ""
    query_ids = [query_id for query_id in qrels if query_id in queries]
    return queries, dict(qrels), query_ids


def iter_corpus_rows(corpus_file: Path) -> Iterator[dict]:
    """Stream corpus rows so large BEIR collections never enter a Python dict."""
    with corpus_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def corpus_line_count(corpus_file: Path) -> int:
    with corpus_file.open("rb") as handle:
        return sum(1 for _ in handle)


def encode_corpus_streaming(
    model,
    corpus_file: Path,
    output_path: Path,
    ids_path: Path,
    prefix: str,
    batch_size: int,
    chunk_size: int = 8192,
) -> tuple[np.ndarray, list[str]]:
    """Encode a JSONL corpus into a disk-backed NumPy array in bounded chunks."""
    document_count = corpus_line_count(corpus_file)
    document_ids: list[str] = []
    embeddings = None
    offset = 0
    rows: list[dict] = []

    for row in iter_corpus_rows(corpus_file):
        rows.append(row)
        if len(rows) < chunk_size:
            continue
        chunk_embeddings = encode(model, rows, prefix, batch_size, side="document")
        if embeddings is None:
            embeddings = np.lib.format.open_memmap(
                output_path,
                mode="w+",
                dtype=np.float32,
                shape=(document_count, chunk_embeddings.shape[1]),
            )
        end = offset + len(rows)
        embeddings[offset:end] = chunk_embeddings
        document_ids.extend(str(item["_id"]) for item in rows)
        offset = end
        rows = []

    if rows:
        chunk_embeddings = encode(model, rows, prefix, batch_size, side="document")
        if embeddings is None:
            embeddings = np.lib.format.open_memmap(
                output_path,
                mode="w+",
                dtype=np.float32,
                shape=(document_count, chunk_embeddings.shape[1]),
            )
        end = offset + len(rows)
        embeddings[offset:end] = chunk_embeddings
        document_ids.extend(str(item["_id"]) for item in rows)
        offset = end

    if embeddings is None or offset != document_count:
        raise RuntimeError(
            f"Streaming corpus count mismatch: encoded {offset}, expected {document_count}"
        )
    embeddings.flush()
    with ids_path.open("w", encoding="utf-8") as handle:
        json.dump({"document_ids": document_ids}, handle)
    return np.load(output_path, mmap_mode="r"), document_ids


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
    document_chunk_size: int = 262_144,
    query_chunk_size: int = 64,
) -> dict[str, dict[str, float]]:
    """Exact top-k retrieval while scanning a memmap once in document chunks."""
    top_k = min(top_k, len(document_ids))
    results: dict[str, dict[str, float]] = {}
    for query_start in range(0, len(query_ids), query_chunk_size):
        query_end = min(query_start + query_chunk_size, len(query_ids))
        query_batch = np.asarray(query_embeddings[query_start:query_end], dtype=np.float32)
        query_count = query_end - query_start
        best_scores = np.full((query_count, top_k), -np.inf, dtype=np.float32)
        best_ids = np.full((query_count, top_k), "", dtype=object)

        for start in range(0, len(document_ids), document_chunk_size):
            end = min(start + document_chunk_size, len(document_ids))
            document_chunk = np.asarray(document_embeddings[start:end], dtype=np.float32)
            scores = query_batch @ document_chunk.T
            local_k = min(top_k, end - start)
            local_indices = np.argpartition(-scores, local_k - 1, axis=1)[:, :local_k]
            local_scores = np.take_along_axis(scores, local_indices, axis=1)
            local_ids = np.asarray(document_ids[start:end], dtype=object)[local_indices]

            merged_scores = np.concatenate((best_scores, local_scores), axis=1)
            merged_ids = np.concatenate((best_ids, local_ids), axis=1)
            keep = np.argpartition(-merged_scores, top_k - 1, axis=1)[:, :top_k]
            best_scores = np.take_along_axis(merged_scores, keep, axis=1)
            best_ids = np.take_along_axis(merged_ids, keep, axis=1)

        for row, query_id in enumerate(query_ids[query_start:query_end]):
            order = np.argsort(-best_scores[row])
            results[query_id] = {
                str(best_ids[row, index]): float(best_scores[row, index])
                for index in order
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
        streaming_corpus = dataset_name in STREAMING_DATASETS
        if streaming_corpus:
            # GenericDataLoader is convenient for small BEIR sets but loads
            # multi-million-document corpora into a Python dictionary. Keep
            # only the test queries/qrels in memory and stream corpus.jsonl.
            queries, qrels, query_ids = load_beir_split_metadata(dataset_dir, "test")
            document_ids = None
            document_rows = None
        else:
            corpus, queries, qrels = GenericDataLoader(
                data_folder=str(dataset_dir)
            ).load(split="test")
            document_ids = list(corpus)
            query_ids = list(queries)
            document_rows = [corpus[doc_id] for doc_id in document_ids]
        query_texts = [queries[query_id] for query_id in query_ids]
        print(
            f"{dataset_name}: "
            f"{corpus_line_count(dataset_dir / 'corpus.jsonl') if streaming_corpus else len(document_ids):,} documents, "
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
                with ids_path.open("r", encoding="utf-8") as handle:
                    cached_ids = json.load(handle)
                document_ids = cached_ids["document_ids"]
                query_ids = cached_ids["query_ids"]
            else:
                print(f"Encoding {dataset_name} with {spec['name']}")
                model = load_encoder(model_key, args.device)
                if streaming_corpus:
                    document_embeddings, document_ids = encode_corpus_streaming(
                        model,
                        dataset_dir / "corpus.jsonl",
                        corpus_path,
                        ids_path,
                        spec["document_prefix"],
                        args.batch_size,
                        chunk_size=args.corpus_chunk_size,
                    )
                else:
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
                if not streaming_corpus:
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
