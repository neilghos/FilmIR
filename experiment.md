Best replacement: TREC Deep Learning
Use the TREC Deep Learning passage-ranking benchmark as the primary benchmark.
It has:

- 8.8M-passage corpus
- Hundreds of thousands of training queries
- Official train/dev qrels
- Blind, deeply judged test queries
- Standard NDCG/MRR evaluation
- Strong BM25, DPR, ColBERT, BERT, and dense-retrieval baselines
  The official MS MARCO/TREC release provides 532k training qrels, development qrels, and held-out TREC test queries with deeper relevance judgments. Official dataset documentation and TREC DL overview.
  This gives us the clean experiment:
  Frozen encoder
  → train FiLM conditioner on train qrels
  → select hyperparameters on dev
  → evaluate once on TREC DL test
  No cross-domain transfer is required.
  Best genuinely different benchmark: TREC CAR
  Use TREC Complex Answer Retrieval as the second benchmark.
  It provides:
- Training data
- Held-out test topics
- Passage-level qrels
- Hierarchical/complex information needs
- Manual graded relevance judgments
- Official TREC evaluation infrastructure
  It is especially suitable because query-conditioned modulation may help when the query expresses a complex information need. TREC CAR official release and TREC CAR overview.
  Optional third benchmark: MIRACL
  MIRACL is a strong option if we want multilingual generalization. It has train/dev/test splits, 78k queries, 18 languages, and over 726k human relevance judgments. MIRACL paper.
  However, we would need a multilingual encoder such as mContriever or multilingual E5.
  What I would use for the paper

1. TREC DL — main supervised benchmark.
2. TREC CAR — complex retrieval benchmark.
3. MIRACL — optional multilingual extension.
4. BRIGHT — evaluation-only stress test, not the main training benchmark; it has only about 1,385 reasoning-intensive queries. BRIGHT.
   The cleanest paper claim becomes:
   Query-conditioned FiLM improves frozen dense retrieval when trained with relevance supervision, across standard passage retrieval and complex-answer retrieval tasks.

That is much more defensible than claiming that a pooled zero-shot adapter transfers across unrelated BEIR domains.
