| Method | Pass@1 | Tools | Steps | Peak | Avg | Final | Edits | Sub. | OOC |
|---|---|---|---|---|---|---|---|---|---|
| **Gemma-4-12B-it** | | | | | | | | | |
| A1 ReAct (no memory) | 0.293 | 71.7 | 71.1 | 35K | 20K | 35K | 0 | 62 | 27 |
| A2 Summarize-on-threshold | 0.273 | 89.1 | 88.2 | 32K | 19K | 26K | 0.58 | 85 | 0 |
| A3 ACM Base (untrained) | \textbf{0.384} | 94.6 | 93.8 | 39K | 20K | 29K | 0.48 | 82 | 3 |
| **MiMo-V2.6-Distill-Qwen-9B** | | | | | | | | | |
| A1 ReAct (no memory) | 0.121 | 143.3 | 129.6 | 57K | 26K | 57K | 0 | 20 | 59 |
| A2 Summarize-on-threshold | 0.212 | 209.3 | 190.5 | 54K | 23K | 27K | 1.54 | 30 | 0 |
| A3 ACM Base (untrained) | \textbf{0.263} | 199.6 | 183.4 | 54K | 23K | 28K | 0.7 | 37 | 2 |
| **Qwen3.5-9B** | | | | | | | | | |
| A1 ReAct (no memory) | 0.172 | 123.7 | 123.4 | 57K | 27K | 57K | 0 | 30 | 66 |
| A2 Summarize-on-threshold | \textbf{0.374} | 176.6 | 175.7 | 54K | 26K | 35K | 1.12 | 46 | 0 |
| A3 ACM Base (untrained) | 0.303 | 182.9 | 182.4 | 54K | 25K | 37K | 1 | 48 | 5 |
| **Reference: ACM paper, Qwen3.5-9B base** | | | | | | | | | |
| \textit{ReAct (ACM paper)} | 0.489 | 74.7 | -- | 59K | -- | -- | -- | -- | -- |
| \textit{ACM Base (ACM paper)} | 0.508 | 77.6 | -- | 46K | -- | -- | -- | -- | -- |
| \textit{ACM Post-Trained (paper)} | 0.530 | 79.3 | -- | 50K | -- | -- | -- | -- | -- |
