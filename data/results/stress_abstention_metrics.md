| Method | Stress Type | N | Qwen Explicit Abstention Rate | Qwen Parse Success Rate | Retrieval-ID Gate Correct Abstention | Citation Gate Correct Abstention |
| --- | --- | --- | --- | --- | --- | --- |
| bm25 | heldout_real_cve | 100 | 0.99 | 1.0 | 1.0 | 1.0 |
| dense | heldout_real_cve | 100 | 0.97 | 1.0 | 1.0 | 1.0 |
| hybrid | heldout_real_cve | 100 | 0.95 | 0.98 | 1.0 | 1.0 |
| hybrid_reranker | heldout_real_cve | 100 | 0.98 | 1.0 | 1.0 | 1.0 |
| bm25 | nonexistent_cve | 50 | 1.0 | 1.0 | 1.0 | 1.0 |
| dense | nonexistent_cve | 50 | 1.0 | 1.0 | 1.0 | 1.0 |
| hybrid | nonexistent_cve | 50 | 1.0 | 1.0 | 1.0 | 1.0 |
| hybrid_reranker | nonexistent_cve | 50 | 1.0 | 1.0 | 1.0 | 1.0 |
| bm25 | gold_evidence_removed | 50 | 0.96 | 1.0 | 1.0 | 1.0 |
| dense | gold_evidence_removed | 50 | 1.0 | 0.98 | 1.0 | 1.0 |
| hybrid | gold_evidence_removed | 50 | 1.0 | 0.98 | 1.0 | 1.0 |
| hybrid_reranker | gold_evidence_removed | 50 | 0.94 | 1.0 | 1.0 | 1.0 |
