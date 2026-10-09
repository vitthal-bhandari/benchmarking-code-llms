# Memory-arm comparison

| Arm | policy | n | edited | edits | vol / forced / harness | mean/inst | peak median | compressed | archive | resolved |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| mem_none_24k_332110 | none | 42 | 0 | 0 | 0 / 0 / 0 | 0.0 | 20,292 | 0 | 0 | — |
| mem_summarize_24k_332111 | summarize | 42 | 42 | 347 | 0 / 0 / 347 | 8.26 | 18,117 | 3,782,074 | 4,413,640 | — |
| mem_acm_24k_332112 | acm | 42 | 38 | 195 | 0 / 195 / 0 | 4.64 | 19,647 | 2,050,884 | 2,299,471 | — |

**A3 voluntary rate:** 0/195 edits (0%) were the model's own choice; the rest were forced at 95% cap. The untrained model rarely self-manages — the gap OPD training would fill.
