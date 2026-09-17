# MER4.0 — Controlled-protocol verification (reproduce registered values)

The `run_cell.sh` now takes an 8th arg `CACHE_STATE` (default `MISS`). `MISS` = clear
`~/.qwen38-mtp/kv-ssd/` + fresh server (full prefill by construction); `RAM-HIT` = fresh
server + one throwaway priming request (guaranteed RAM hit). Per-rep recording: `cache_state`
(controlled) + `cache_ssd_promoted` (verified from the server's `Lazy-loaded radix prefix`
log line).

| Cell | Metric | Registered (MER2) | MER4.0 (controlled MISS) | Verdict |
|------|--------|-------------------|--------------------------|---------|
| essay decode | ttlt (tok/s) | 24.99 | **25.65** | REPRODUCES (Δ+2.6%) |
| essay decode | tEvalAvg (ms) | 80.7 | **81.6** | REPRODUCES (Δ+1.1%) |
| essay decode | stream hash | `949b9423…` | `949b9423…` | EXACT |
| 32K prefill | prefill wall (s) | 125.50 | **132.5** | REPRODUCES (Δ+5.6%) |
| 32K prefill | prompt tokens | 32780 | 32780 | EXACT |
| 32K prefill | cache state | MISS (full prefill) | MISS (ssd_promoted=false) | EXACT |

All deltas within cross-rep/cross-session variance. The controlled-MISS protocol is sound.

## 32K PF2 phase breakdown (controlled MISS, v0.32.2)

```
total_ms=132445.71  ffn_ms=62710.50 (47.3%)  gdn_ms=29232.19 (22.1%)
attn_ms=38110.22 (28.8%, incl sdpa 29963.88)  qkv_ms=4791.95 (3.6%)
oproj_ms=2706.36 (2.0%)  rope_ms=638.78 (0.5%)  norm_ms=1485.83 (1.1%)  residual_ms=906.98
```

FFN is 47.3% of prefill (consistent with the FFP1 ~43–49% band); causal attention 28.8%.
