| Method | Pass@1 | Tools | Peak Tok. | Edits | Vol. | Util. | Ovh. | Steps | Sub. | OOC |
|---|---|---|---|---|---|---|---|---|---|---|
| **Gemma-4-12B-it** | | | | | | | | | | |
| A1 ReAct (no memory) | 0.293 | 71.7 | 35K | 0 | 0 | 0.561 | 0 | 71.1 | 62 | 27 |
| A2 Summarize-on-threshold | 0.273 | 89.1 | 32K | 0.58 | 0 | 0.549 | 0.0087 | 88.2 | 85 | 0 |
| A3 ACM Base (untrained) | 0.384 | 94.6 | 39K | 0.48 | 0 | 0.58 | 0.0084 | 93.8 | 82 | 3 |
| **MiMo-V2.6-Distill-Qwen-9B** | | | | | | | | | | |
| A1 ReAct (no memory) | 0.121 | 143.3 | 57K | 0 | 0 | 0.713 | 0 | 129.6 | 20 | 59 |
| A2 Summarize-on-threshold | 0.212 | 209.3 | 54K | 1.54 | 0 | 0.646 | 0.0081 | 190.5 | 30 | 0 |
| A3 ACM Base (untrained) | 0.263 | 199.6 | 54K | 0.7 | 0 | 0.674 | 0.0059 | 183.4 | 37 | 2 |
| **Qwen3.5-9B** | | | | | | | | | | |
| A1 ReAct (no memory) | 0.172 | 123.7 | 57K | 0 | 0 | 0.805 | 0 | 123.4 | 30 | 66 |
| A2 Summarize-on-threshold | 0.374 | 176.6 | 54K | 1.12 | 0 | 0.787 | 0.0077 | 175.7 | 46 | 0 |
| A3 ACM Base (untrained) | 0.303 | 182.9 | 54K | 1 | 0 | 0.771 | 0.0068 | 182.4 | 48 | 5 |
| **Reference: ACM paper, Qwen3.5-9B base** | | | | | | | | | | |
| \textit{ReAct (ACM paper)} | 0.489 | 74.7 | 59K | -- | -- | -- | -- | -- | -- | -- |
| \textit{ACM Base (ACM paper)} | 0.508 | 77.6 | 46K | -- | -- | -- | -- | -- | -- | -- |
| \textit{ACM Post-Trained (paper)} | 0.530 | 79.3 | 50K | -- | -- | -- | -- | -- | -- | -- |
