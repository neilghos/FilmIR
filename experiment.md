# Experiment Plan

## Goal

Test whether a query-conditioned FiLM adapter improves frozen dense retrieval
representations across retrieval backbones and datasets.

Metric: NDCG@10 over the full BEIR corpus.

## Result tables

### Table 1: Full supervised in-domain adaptation

Train one FiLM adapter per dataset using the official training qrels. Use the
development split for selection when available and evaluate once on test.

Datasets:

- MSMARCO, using a fixed 2,000--3,000-example training subset.
- NFCorpus
- NQ
- HotpotQA
- FiQA
- FEVER
- SciFact

For datasets without an official development split, create a fixed validation
split from the training queries. The MSMARCO subset and all validation splits
must be fixed before test evaluation.

### Table 2: Zero-shot cross-dataset transfer

Train one pooled FiLM adapter on the training data from source datasets, then
evaluate on a held-out target dataset whose training qrels were not used.
Use leave-one-dataset-out evaluation where practical.

This is true zero-shot transfer for the adapter. A pooled adapter trained on
all datasets and evaluated on those same datasets is multi-dataset supervised
adaptation, not zero-shot; report it separately if included.

### Table 3: Few-shot target adaptation

Start from the pooled source-only adapter and adapt it using a small target
support set. Evaluate on untouched target test queries.

Support sizes:

```text
k in {2, 4, 8, 16, 32}
```

Repeat support sampling over multiple seeds and report mean and standard
deviation. Support queries must not appear in the evaluation set.

## Baseline systems

Core systems:

- BM25: standalone sparse lexical baseline.
- MiniLM: compact general-purpose dense encoder.
- MultiQA: QA-specialized dense encoder.
- BGE: modern general-purpose dense encoder.
- E5: retrieval-specialized contrastive encoder with query/passage prefixes.
- Contriever: unsupervised contrastive dense encoder.
- DPR: classic supervised QA dual encoder with separate question and passage
  encoders.

SPLADE is excluded from the FiLM experiments because it is a sparse neural
retriever and is not directly compatible with the current dense FiLM adapter.
It may be reported only as a standalone sparse baseline if needed.

## FiLM configuration

Use one globally fixed configuration before the main comparison:

```text
parameterization: polar
hidden dimension: 128
learning rate: 1e-4
modulation scale: 0.25
score alpha: 0.5
modulation regularization: 0.02
epochs: 100
```

FiLM is trained while the base query/document encoder remains frozen. Report
both the mixed score and FiLM-only score against the frozen baseline.

Do not tune hyperparameters against test NDCG. Use development data or fixed
training-only validation splits, then lock the configuration across models and
datasets.
