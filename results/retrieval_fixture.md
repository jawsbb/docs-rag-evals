# Retrieval eval: `fixture` corpus

- Corpus: 24 pages / 197 chunks, upstream commit `311e2c721c1e`
- Golden set: 45 answerable questions scored (+6 unanswerable, reported separately)
- Embeddings: `BAAI/bge-small-en-v1.5`; generated 2026-11-16 14:20 UTC

## Overall (page level)

| variant | recall@1 | recall@3 | recall@5 | recall@10 | mrr | ndcg@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| fts | 0.544 | 0.722 | 0.844 | 0.978 | 0.717 | 0.767 | 2.2 | 4.5 |
| **vector** (gated) | 0.811 | 0.978 | 0.978 | 1.000 | 0.926 | 0.947 | 5.4 | 6.2 |
| hybrid | 0.667 | 0.933 | 1.000 | 1.000 | 0.839 | 0.871 | 8.2 | 11.0 |
| hybrid_rerank | 0.911 | 0.967 | 1.000 | 1.000 | 0.985 | 0.983 | 1009.2 | 1033.7 |

## recall@5 by question type

| type | n | fts | vector | hybrid | hybrid_rerank |
|---|---|---|---|---|---|
| how_to | 10 | 0.900 | 1.000 | 1.000 | 1.000 |
| keyword | 8 | 0.750 | 0.875 | 1.000 | 1.000 |
| lookup | 10 | 1.000 | 1.000 | 1.000 | 1.000 |
| multi_hop | 7 | 0.857 | 1.000 | 1.000 | 1.000 |
| paraphrase | 10 | 0.700 | 1.000 | 1.000 | 1.000 |

## Unanswerable questions (not gated)

Top-1 cosine similarity of the vector retriever. A usable no-answer threshold needs the unanswerable values to sit clearly below the answerable mean.

- answerable mean: 0.7634
- q065: 0.6685
- q066: 0.7129
- q067: 0.6742
- q068: 0.69
- q069: 0.6734
- q070: 0.7787

## `vector` misses (recall@10 < 1): 0

