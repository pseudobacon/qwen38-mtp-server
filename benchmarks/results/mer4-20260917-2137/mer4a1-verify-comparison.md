# MER4A.1 — Verify-width M=1..9 (routed vs fallback), v0.32.2 vs v0.31.1

Sustained throughput, `wide_global` (the key QMV win metric). v0.31.1 baseline =
`benchmarks/results/w5-throughput.txt` (registered, M ∈ {1,2,4,8,9}); v0.32.2 =
`mer4a1-verify-m1-9.json` (M ∈ {1..9}).

| M | v311 routed | v311 fallbk | v311 win | v322 routed | v322 fallbk | v322 win |
|---|------------|------------|----------|------------|------------|----------|
| 1 | 311.8 | 330.5 | 0.94x | 371.5 | 384.3 | 0.97x |
| 2 | 293.5 | 248.5 | 1.18x | 296.7 | 246.8 | 1.20x |
| 4 | 201.6 | 134.9 | 1.49x | 196.1 | 128.6 | 1.53x |
| 8 | 101.8 | 70.8 | 1.44x | 101.6 | 66.9 | 1.52x |
| 9 | 88.5 | 58.0 | 1.53x | 86.7 | 55.8 | 1.55x |

**Finding:** the M=1..9 win profile is UNCHANGED on v0.32.2 — M=1 is a wash (0.94–0.97x),
M=2..9 routed wins (1.18–1.55x). qmv_wide / the M5 batch limit did NOT change the win
profile. The QMV dispatch threshold (routed at M=2..9) does NOT need re-tuning.
