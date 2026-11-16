# Retrieval eval: `fixture` corpus

- Corpus: 24 pages / 197 chunks, upstream commit `311e2c721c1e`
- Golden set: 44 answerable questions scored (+6 unanswerable, reported separately)
- Embeddings: `BAAI/bge-small-en-v1.5`; generated 2026-11-16 13:48 UTC

## Overall (page level)

| variant | recall@1 | recall@3 | recall@5 | recall@10 | mrr | ndcg@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| fts | 0.545 | 0.727 | 0.841 | 0.977 | 0.711 | 0.764 | 2.5 | 4.5 |
| **vector** (gated) | 0.830 | 0.977 | 0.977 | 1.000 | 0.936 | 0.953 | 6.1 | 6.7 |
| hybrid | 0.682 | 0.943 | 1.000 | 1.000 | 0.846 | 0.876 | 8.6 | 11.1 |
| hybrid_rerank | 0.932 | 0.977 | 1.000 | 1.000 | 1.000 | 0.993 | 1116.8 | 1189.8 |

## recall@5 by question type

| type | n | fts | vector | hybrid | hybrid_rerank |
|---|---|---|---|---|---|
| how_to | 10 | 0.900 | 1.000 | 1.000 | 1.000 |
| keyword | 8 | 0.750 | 0.875 | 1.000 | 1.000 |
| lookup | 10 | 1.000 | 1.000 | 1.000 | 1.000 |
| multi_hop | 6 | 0.833 | 1.000 | 1.000 | 1.000 |
| paraphrase | 10 | 0.700 | 1.000 | 1.000 | 1.000 |

## Unanswerable questions (not gated)

Top-1 cosine similarity of the vector retriever. A usable no-answer threshold needs the unanswerable values to sit clearly below the answerable mean.

- answerable mean: 0.7652
- q065: 0.6685
- q066: 0.7129
- q067: 0.6742
- q068: 0.69
- q069: 0.6734
- q070: 0.7787

## `vector` misses (recall@10 < 1): 0

