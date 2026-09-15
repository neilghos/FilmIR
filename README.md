# DBpedia-Entity Node2Vec + FiLM

## BEIR dense baselines

Run the first-stage baseline sweep on the configured standard BEIR corpora.
The dense backbones currently include MiniLM, MultiQA, BGE, E5, Contriever,
and DPR.

```bash
python -m pip install -r requirements-baseline.txt
python run_beir_baselines.py
```

Embeddings are cached under `runs/beir_cache/`; metrics and top-1000 runs are
written under `runs/beir_baselines/`. The runner uses the test split only for
baseline evaluation. Its cached embeddings are the input to the later FiLM
training stage.

## Train FiLM adapters

After the baseline cache exists, train one FiLM adapter per dataset and base
retriever:

```bash
python train_film.py
```

FiLM checkpoints, metrics, and runs are written under `runs/beir_film/`.
NFCorpus and SciFact use their training split and evaluate on test. ArguAna
and SciDocs use five-fold query cross-validation because they have no standard
training qrels. Select one fold with `--fold 0` through `--fold 4`.

Start with one small smoke test:

```bash
python run_beir_baselines.py --datasets nfcorpus --models minilm
python train_film.py --datasets nfcorpus --models minilm --epochs 5
```

Contriever and DPR use Hugging Face encoders. DPR encodes queries and
documents with its separate question and context encoders. Both can be used
as frozen baselines and as FiLM backbones:

```bash
python run_beir_baselines.py --datasets nfcorpus --models contriever dpr
python train_film.py --datasets nfcorpus --models contriever dpr
```

The trainer uses bounded FiLM residuals and a BPR objective. Its final score
is the baseline score plus a small FiLM correction. Each
`runs/beir_film/<dataset>/<model>/metrics.json` contains `baseline`,
`film_mixed`, and `film_only` metrics. The matching run files contain the
top-1000 rankings.

Use `--film-parameterization polar` to test the bounded polar variant, which
constrains `(gamma - 1, beta)` to lie inside a disk while preserving the
identity initialization.

The first hypothesis test is:

```text
fixed dense retrieval  vs  query-conditioned dense retrieval
```

## STaRK KG/text retrieval path

The STaRK pipeline is separate from the pure BEIR IR experiments. It loads
official STaRK query splits together with the semi-structured KG: typed node
IDs, typed edges, candidate-entity IDs, and lazy node-text access. Install its
optional dependencies with:

```bash
python -m pip install -r requirements-stark.txt
python inspect_stark.py --dataset prime --split train
```

The loader stores STaRK data under `data/stark/`, never under `data/beir/`.
The first structural baseline uses the internal contiguous node indices (not
external biomedical IDs) as learnable LightGCN node IDs. It trains a frozen
MiniLM query encoder projection with full-candidate multi-positive softmax:

```bash
python train_stark_lightgcn.py \
  --dataset prime \
  --epochs 200 \
  --eval-every 40 \
  --embedding-dim 128 \
  --layers 2 \
  --output-dir runs/stark_prime_lightgcn
```

This baseline uses graph connectivity but ignores relation labels. The text
feature branch and FiLM conditioning will be added separately.

## TREC CAR query-conditioned retrieval path

TREC CAR is kept separate from BEIR because it uses official CBOR outlines,
paragraph IDs, section-path queries, and trec_eval qrels. Install the official
reader with:

```bash
python -m pip install -r requirements-trec-car.txt
```

The loader is lazy over the large paragraph corpus and eagerly reads only the
outline queries and qrels:

```bash
python inspect_trec_car.py \
  --outlines /path/to/benchmarkY1-test.cbor.outlines \
  --qrels /path/to/benchmarkY1-test.cbor.hierarchical.qrels \
  --paragraphs /path/to/paragraphCorpus/dedup.articles-paragraphs.cbor \
  --split benchmarkY1test
```

Use matching TREC CAR release files and choose one qrel policy (`hierarchical`,
`toplevel`, or `tree`) for an experiment. The loader preserves official query
and paragraph IDs, creates query text from the clean section path, and never
materializes the full paragraph corpus unless explicitly requested.

Train the frozen dense encoder plus FiLM adapter by supplying matching
official train/evaluation outlines and qrels. The paragraph corpus is shared
by both splits:

```bash
python train_trec_car_film.py \
  --train-outlines /path/to/train.cbor.outlines \
  --train-qrels /path/to/train.cbor.hierarchical.qrels \
  --eval-outlines /path/to/benchmarkY1-test.cbor.outlines \
  --eval-qrels /path/to/benchmarkY1-test.cbor.hierarchical.qrels \
  --paragraphs /path/to/paragraphCorpus/dedup.articles-paragraphs.cbor \
  --model minilm \
  --film-parameterization polar \
  --output-dir runs/trec_car_film/minilm
```

The trainer uses baseline-ranked hard negatives plus random negatives and
writes `baseline.run`, `film_mixed.run`, and `film_only.run` in trec_eval run
format. `metrics.json` contains diagnostic NDCG@10, Recall@100, and MRR; use
the official `trec_eval` command for final reported metrics. Add
`--max-documents 10000` for a bounded smoke test. Omitting that option streams
and encodes the complete paragraph corpus for the specified release.

Run frozen dense baselines with the standalone verification runner:

```bash
python run_trec_car_baselines.py \
  --outlines /path/to/benchmarkY3test.public.cbor.outlines \
  --qrels /path/to/trec-car-benchmarkY3test-section.qrels \
  --paragraphs /path/to/paragraphCorpus/dedup.articles-paragraphs.cbor \
  --models minilm bge e5 contriever \
  --split benchmarkY3test \
  --output-dir runs/trec_car_baselines/y3
```

The runner reports diagnostic MAP, MRR, R-precision, NDCG@10, and Recall@100
and writes one standard TREC run per encoder. To verify an Anserini/Pyserini
BM25 run without changing its scores, pass it directly:

```bash
python run_trec_car_baselines.py \
  --qrels /path/to/trec-car-benchmarkY3test-section.qrels \
  --run-file /path/to/bm25.run \
  --output-dir runs/trec_car_baselines/y3/bm25
```

## Cross-dataset zero-shot transfer

Train one pooled MiniLM FiLM adapter on 5,000 training queries from each of
six source datasets, then evaluate it on held-out target datasets without
using their queries or qrels for training:

```bash
python train_transfer.py \
  --source-datasets msmarco fiqa fever hotpotqa nfcorpus scifact \
  --source-queries 5000 \
  --target-datasets arguana scidocs \
  --model minilm
```

This requests 30,000 source queries in total. NFCorpus contributes its full
positive training-query set (2,590 queries), because it has fewer than 5,000
such queries. Each source corpus is streamed and capped by
`--source-document-pool` (100,000 documents by default), so the full source
corpora are not embedded. The target datasets are evaluated with their test
qrels only. Results are written under
`runs/transfer_film/pooled_6x5000/minilm/`.

For the original single-source experiment, pass `--source-dataset msmarco`
instead of `--source-datasets ...`; its output path becomes
`runs/transfer_film/pooled_1x5000/minilm/`. A source dataset cannot also be
listed as a zero-shot target; use leave-one-dataset-out runs when evaluating
transfer to one of the six source domains.

## KG experiments

The experimental KG/STaRK code is isolated under `KGstark/`. The root
retrieval pipeline uses the frozen base retriever's query embedding directly
as input to FiLM; it does not instantiate a separate BERT query encoder.
