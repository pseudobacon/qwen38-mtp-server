# MER2 A/B Analysis — mlx-v0322-merge-20260917-1831

## Decode throughput (ttlt, tok/s) — rep 1 discarded as warmup, 5 measured
### essay
- incumbent: reps=['25.21', '22.51', '23.81', '23.24', '23.56', '22.97'], measured(mean r2-6)=23.22
- upgraded:  reps=['25.66', '26.04', '25.85', '25.18', '24.20', '23.69'], measured(mean r2-6)=24.99
- paired deltas (r2-6, %): ['+15.69', '+8.56', '+8.35', '+2.69', '+3.15']
- mean delta: +7.688%

### specdec
- incumbent: reps=['26.33', '24.34'], measured(mean r2-6)=24.34
- upgraded:  reps=['26.75', '25.70'], measured(mean r2-6)=25.70
- paired deltas (r2-6, %): ['+5.56']
- mean delta: +5.563%

## Determinism (stream hash)
- essay-inc: ['949b9423bd85', '949b9423bd85', '949b9423bd85', '949b9423bd85', '949b9423bd85', '949b9423bd85']
- essay-upg: ['949b9423bd85', '949b9423bd85', '949b9423bd85', '949b9423bd85', '949b9423bd85', '949b9423bd85']
- specdec-inc: ['139acb9d30fe', '06882d856267']
- specdec-upg: ['139acb9d30fe', '139acb9d30fe']

## DETERMINISM GATE: FAIL (incumbent specdec non-deterministic)
- specdec-inc: r1=139acb9d (registry), r2=06882d85 (diverged); 8/9 total runs = 06882d85
- specdec-upg: all = 139acb9d (deterministic)
- essay: both sides fully deterministic (949b9423)
- First-divergence inc(0688) vs upg(139a): char 4847/5150 (94% through), similarity 94.5% — knife-edge near end

## VERDICT: STOP — do not merge (determinism gate triggered on incumbent side)
The incumbent (v0.31.6) is non-deterministic on specdec (pre-existing). The upgraded (v0.32.2) is
deterministic and matches the registry. Per task gate: 'If determinism fails on either side, STOP.'