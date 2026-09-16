# Metrics — `/metrics` surface

> Provenance: the "Radix KV prefix reuse" section below was ported from the
> recovered lineage (`qwen-mtp-server` @ `efcf595`, commit `820c9f4`'s
> `docs/metrics.md` addition, 2026-09-16); the canonical repo had no
> `docs/metrics.md` before this port.

## Radix KV prefix reuse (Stage 1)

Per-request, carried on `RequestMetrics` and yielded through the generation
stream:

- `matchedPrefixTokens`: prefix tokens matched in the Radix tree (0 = no prior
  cache).
- `reusedPrefixTokens`: prefix tokens actually reused (0 on a desync fallback
  where `trimmableOffset != prefixCount`). Always `<= matchedPrefixTokens`.
- `radixPrefillSkipped`: 1 when the whole prompt was reused (reused prefix ==
  full prompt, prefill skipped), else 0.

Aggregate on `/metrics` (`MetricsSummary`, snake_case, over the rolling window):

- `prefix_reuse_hits`: requests that adopted a cached prefix
  (`reusedPrefixTokens > 0`).
- `prefix_reuse_fallbacks`: requests that matched a prefix count but fell back
  to full prefill (`matchedPrefixTokens > 0` and `reusedPrefixTokens == 0`).
- `prefix_reuse_tokens_saved`: total prompt tokens not re-prefilled thanks to
  reuse.
