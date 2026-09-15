# FiLM Retrieval Experiment Plan

## Table 1: 8-shot adaptation

Train the FiLM conditioner on eight labeled query--document examples from
the fixed adaptation split, then evaluate on the untouched BEIR test split.
For test-only datasets, keep the eight sampled test queries in the evaluation
denominator but assign them empty runs, following Promptagator's zero-credit
protocol.

## Table 2: Full supervised adaptation

Train the FiLM conditioner on all labeled queries from the fixed adaptation
split, then evaluate on the BEIR test split.

Datasets for both tables:

```text
NFCorpus       dev
SciFact        train
FiQA           dev
FEVER          dev
HotpotQA       dev
DBPedia        dev
```

The frozen encoder, document embeddings, evaluator, and test corpus remain
unchanged between the two settings. The only difference is the number of
adaptation queries.
