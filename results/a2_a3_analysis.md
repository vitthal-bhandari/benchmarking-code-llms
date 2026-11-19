## 1. Noise floor: same arm run twice (Qwen3.5-9B, 64k, step 250 vs 500)

| arm | Pass@1 run1 | Pass@1 run2 | flips 1->0 | flips 0->1 | discordant | McNemar p |
|---|--:|--:|--:|--:|--:|--:|
| A1 none | 0.172 | 0.192 | 6 | 8 | 14 (14%) | 0.79 |
| A2 summarize | 0.374 | 0.364 | 16 | 15 | 31 (31%) | 1.00 |
| A3 acm | 0.303 | 0.384 | 10 | 18 | 28 (28%) | 0.18 |

## 2. A2 vs A3 per cell, split by whether the memory mechanism ever fired

| cell | n | A2 | A3 | A3-A2 | 95% CI | McNemar p | discordant | never-compressed: n, A2, A3 | compressed: n, A2, A3 |
|---|--:|--:|--:|--:|---|--:|--:|---|---|
| qwen 64k s250 | 99 | 37 | 30 | -0.071 | [-0.182, +0.040] | 0.28 | 31 | 17: 13 vs 13 | 82: 24 vs 17 |
| qwen 64k s500 | 99 | 36 | 38 | +0.020 | [-0.091, +0.121] | 0.85 | 28 | 25: 16 vs 14 | 74: 20 vs 24 |
| qwen 32k s500 | 99 | 36 | 27 | -0.091 | [-0.172, -0.010] | 0.05 | 17 | 0: 0 vs 0 | 99: 36 vs 27 |
| qwen 128k s500 | 99 | 39 | 43 | +0.040 | [-0.061, +0.141] | 0.57 | 28 | 73: 35 vs 36 | 26: 4 vs 7 |
| mimo 64k s250 | 99 | 21 | 26 | +0.051 | [-0.061, +0.162] | 0.46 | 29 | 23: 14 vs 12 | 76: 7 vs 14 |
| gemma 64k s250 | 99 | 27 | 38 | +0.111 | [+0.030, +0.192] | 0.01 | 17 | 50: 24 vs 29 | 49: 3 vs 9 |

## 3. A3-only events on instances where the arms disagree (Qwen 64k s500)

| instance | won by | A2 exit | A2 edits | A3 exit | A3 edits | A3 nudges | A3 stop | A3 tools |
|---|---|---|--:|---|--:|--:|---|---|
| astropy__astropy-12907 | A3 | Submitted | 0 | Submitted | 1 | 4 | True | {'bash': 292, 'manage_context': 1} |
| astropy__astropy-14309 | A2 | Submitted | 0 | Submitted | 0 | 0 | False | {'bash': 78} |
| astropy__astropy-14508 | A2 | Submitted | 1 | Submitted | 0 | 0 | False | {'bash': 49} |
| astropy__astropy-7336 | A2 | Submitted | 0 | Submitted | 2 | 2 | False | {'bash': 469, 'manage_context': 2} |
| django__django-10880 | A3 | Submitted | 2 | Submitted | 3 | 3 | False | {'bash': 458, 'manage_context': 3} |
| django__django-10973 | A3 | Submitted | 2 | Submitted | 1 | 1 | False | {'bash': 448, 'manage_context': 1} |
| django__django-11066 | A3 | Submitted | 0 | Submitted | 0 | 0 | False | {'bash': 99} |
| django__django-11095 | A3 | Submitted | 1 | Submitted | 1 | 1 | False | {'bash': 133, 'manage_context': 1} |
| django__django-11119 | A3 | LimitsExceeded | 2 | Submitted | 0 | 0 | False | {'bash': 47} |
| django__django-11179 | A3 | LimitsExceeded | 6 | Submitted | 0 | 0 | False | {'bash': 102} |
| django__django-11206 | A2 | Submitted | 3 | Submitted | 2 | 2 | False | {'bash': 193, 'manage_context': 2} |
| django__django-11211 | A3 | LimitsExceeded | 7 | Submitted | 3 | 3 | False | {'bash': 287, 'manage_context': 3} |
| django__django-11265 | A3 | Submitted | 4 | Submitted | 1 | 1 | False | {'bash': 185, 'manage_context': 1} |
| django__django-11292 | A2 | Submitted | 0 | LimitsExceeded | 4 | 4 | False | {'bash': 496, 'manage_context': 4} |
| django__django-11299 | A2 | Submitted | 1 | Submitted | 3 | 3 | False | {'bash': 403, 'manage_context': 3} |
| django__django-11333 | A3 | Submitted | 1 | Submitted | 2 | 2 | False | {'bash': 310, 'manage_context': 2} |
| django__django-11451 | A2 | Submitted | 0 | Submitted | 0 | 0 | False | {'bash': 69} |
| django__django-11551 | A3 | Submitted | 3 | Submitted | 0 | 0 | False | {'bash': 131} |
| django__django-11603 | A2 | Submitted | 0 | Submitted | 0 | 0 | False | {'bash': 128} |
| django__django-11740 | A2 | Submitted | 3 | LimitsExceeded | 4 | 5 | False | {'bash': 496, 'manage_context': 4} |
| django__django-11951 | A2 | Submitted | 1 | Submitted | 0 | 0 | False | {'bash': 55} |
| django__django-11964 | A3 | Submitted | 3 | Submitted | 1 | 1 | False | {'bash': 231, 'manage_context': 1} |
| django__django-12209 | A2 | Submitted | 3 | ContextWindowExceededError | 0 | 1 | False | {'bash': 53} |
| django__django-12304 | A3 | Submitted | 1 | Submitted | 1 | 1 | False | {'bash': 133, 'manage_context': 1} |
| django__django-12663 | A2 | Submitted | 4 | LimitsExceeded | 3 | 3 | False | {'bash': 497, 'manage_context': 3} |
| django__django-12708 | A2 | Submitted | 2 | LimitsExceeded | 2 | 2 | False | {'bash': 498, 'manage_context': 2} |
| django__django-12774 | A3 | Submitted | 0 | Submitted | 2 | 2 | False | {'bash': 224, 'manage_context': 2} |
| django__django-13089 | A3 | LimitsExceeded | 7 | Submitted | 2 | 2 | False | {'bash': 212, 'manage_context': 2} |
