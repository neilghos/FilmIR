# Query-Conditioned Retrieval Plan

## Stage 1 — BEIR dense-retrieval baseline + FiLM

Load the four smallest BEIR corpora—NFCorpus, SciFact, ArguAna, and SciDocs—through BEIR. Evaluate several pretrained dense retrievers and cache their query/entity embeddings.

Train an NLP FiLM generator on the training folds while keeping the baseline entity embeddings fixed. Re-evaluate the FiLM-conditioned retriever on held-out queries.

```text
pretrained retriever → baseline embeddings → baseline evaluation
                                      ↓
                         query-conditioned FiLM
                                      ↓
                              held-out evaluation
```

## Stage 2 — Graph embedding retrieval

Replace supervised semantic embeddings with Node2Vec/RDF2Vec embeddings learned from KG structure. Compare normal retrieval against FiLM-conditioned retrieval.

## Stage 3 — Learned KG encoder

Replace Node2Vec with a learned graph or graph-text encoder that produces entity embeddings, then use the same FiLM-conditioned retrieval head.

## Stage 4 — General graphified IR retrieval

Convert arbitrary IR datasets into document/entity graphs, learn graph representations, and apply the same query-conditioned FiLM retrieval framework.

## Core comparison

```text
supervised embedding + normal retrieval
supervised embedding + FiLM retrieval
graph embedding + normal retrieval
graph embedding + FiLM retrieval
```
