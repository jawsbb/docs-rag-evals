# Retrieval eval: `full` corpus

- Corpus: 239 pages / 1672 chunks, upstream commit `311e2c721c1e`
- Golden set: 64 answerable questions scored (+6 unanswerable, reported separately)
- Embeddings: `BAAI/bge-small-en-v1.5`; generated 2026-11-16 13:46 UTC

## Overall (page level)

| variant | recall@1 | recall@3 | recall@5 | recall@10 | mrr | ndcg@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| fts | 0.273 | 0.453 | 0.578 | 0.688 | 0.420 | 0.473 | 15.1 | 31.3 |
| **vector** (gated) | 0.734 | 0.875 | 0.922 | 0.961 | 0.859 | 0.872 | 10.3 | 12.4 |
| hybrid | 0.477 | 0.656 | 0.812 | 0.938 | 0.646 | 0.701 | 29.1 | 43.8 |
| hybrid_rerank | 0.742 | 0.906 | 0.938 | 0.945 | 0.875 | 0.878 | 1079.2 | 1167.3 |

## recall@5 by question type

| type | n | fts | vector | hybrid | hybrid_rerank |
|---|---|---|---|---|---|
| how_to | 14 | 0.643 | 1.000 | 0.786 | 1.000 |
| keyword | 12 | 0.417 | 0.833 | 0.833 | 0.833 |
| lookup | 14 | 0.857 | 1.000 | 1.000 | 1.000 |
| multi_hop | 9 | 0.444 | 0.889 | 0.667 | 0.889 |
| paraphrase | 15 | 0.467 | 0.867 | 0.733 | 0.933 |

## Unanswerable questions (not gated)

Top-1 cosine similarity of the vector retriever. A usable no-answer threshold needs the unanswerable values to sit clearly below the answerable mean.

- answerable mean: 0.7798
- q065: 0.8527
- q066: 0.7588
- q067: 0.7558
- q068: 0.8502
- q069: 0.7123
- q070: 0.7787

## `vector` misses (recall@10 < 1): 3

- **q036** (keyword) What is `SB_EXECUTION_ID` in a hosted Edge Function?
  - expected: functions/secrets; got top 3: functions/error-codes, functions/architecture, functions/regional-invocation
- **q061** (paraphrase) Using an AWS SDK straight from the browser, how do I limit each signed-in user to the files my access policies allow?
  - expected: storage/s3/authentication; got top 3: storage/quickstart, storage/security/access-control, auth/auth-anonymous
- **q064** (multi_hop) Creating an index on a big table from the SQL Editor times out and blocks writes. How should I run it?
  - expected: database/postgres/indexes, database/postgres/timeouts; got top 3: database/postgres/indexes, database/extensions/index_advisor, database/orioledb
