# DBpedia-Entity Node2Vec + FiLM

## BEIR dense baselines

Run the first-stage baseline sweep on the four smallest standard BEIR
corpora: NFCorpus, SciFact, ArguAna, and SciDocs.

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

## Legacy DBpedia/Node2Vec path

The following path is Stage 2 of the plan and is separate from the BEIR
adapter above.

### Data layout

Download and unpack the BEIR `dbpedia-entity` dataset into a directory with:

```text
data/dbpedia-entity/
  corpus.jsonl
  queries.jsonl
  qrels/test.tsv
```

The corpus must be the BEIR DBpedia-Entity corpus. Its entity IDs must match
the qrels. The graph used for Node2Vec is a separate edge list; it must use
the same entity IDs.

### Run the loader check

```bash
python - <<'PY'
from data_loader import load_dbpedia_entity

data = load_dbpedia_entity("data/dbpedia-entity", split="test")
print(data.num_entities, len(data.queries), len(data.qrels))
print(data.entity_ids[0])
PY
```

### Train the FiLM model

`Z_matrix.pt` must have one row per corpus entity, in exactly the same order as
`corpus.jsonl`.

```bash
python trainer.py --data-dir data/dbpedia-entity --z-matrix Z_matrix.pt \
  --mode baseline --checkpoint baseline.pt

python trainer.py --data-dir data/dbpedia-entity --z-matrix Z_matrix.pt \
  --mode film --checkpoint film.pt
```

The default training split is `dev`; keep `test` held out until the evaluator
is wired.  Pass `--split test` only for a deliberate smoke test.

The current trainer uses the gamma-only form of FiLM because beta is a
candidate-independent additive term under dot-product retrieval. The normal
retrieval baseline is selected with `--mode baseline`; FiLM is selected with
`--mode film`.

### Train Node2Vec with the same entity ordering

Provide a tab-separated graph edge list using the same DBpedia entity IDs as
`corpus.jsonl`. Two-column and three-column (`head`, `relation`, `tail`) files
are accepted:

```bash
python node2vec.py \
  --data-dir data/dbpedia-entity \
  --edges data/dbpedia_edges.tsv \
  --output Z_matrix.pt \
  --node-ids node_ids.json
```
