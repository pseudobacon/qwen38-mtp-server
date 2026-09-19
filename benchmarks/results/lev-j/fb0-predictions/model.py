#!/usr/bin/env python3
"""FB0: re-derive the flash + pc compounding model with CURRENT measured numbers.

Replaces the LEV-D arithmetic (which used the STALE c_dense=753.9 from an older
MLX) with the in-session current-MLX measurements:
  - 32K pc=2048 full prefill (PF2, this run): total=110.4s, SDPA=17.0s (15.4%),
    FFN=57.5s, GDN=26.0s, other=9.8s  (L=32780 tok)
  - FA micro-bench (single-shot median, n=20): flash vs incumbent dense SDPA
    per full-prefill token:  32K  18.4 vs 64.8 us/tok  (3.51x)
                                      64K  39.8 vs 143.2 us/tok (3.60x)

The flash kernel replaces ONLY the SDPA term; FFN/GDN/other are pc-dependent
(FFN GEMM per-token rises with pc via the MCP1 M-curve) but flash-independent.

Win condition (per FD): total(flash,pc) < total(dense,pc=2048) at the same L.
KEEP gate (original): >=10% mean 32K prefill wall at the winning (flash,pc) cell.
"""

# --- current measured 32K pc=2048 (PF2, s) -------------------------------
L32, L64 = 32780, 65548
total32 = 110403.51 / 1e3          # s
sdpa32  = 16972.59 / 1e3
ffn32   = 57542.34 / 1e3
gdn32   = 26035.18 / 1e3
other32 = (23226.85 - 16972.59 + 2794.07 + 778.62) / 1e3   # attn-nonsdpa+norm+resid

# 64K: SDPA scales O(L^2) ~ 2x per-token vs 32K; non-SDPA ~ O(L) (registered 249s)
sdpa64 = sdpa32 * (L64 / L32) * 2.0          # O(L^2): per-token 2x, total 2x*... 
# Actually SDPA total ~ L^2 (per-token ~ L): 64K total = 32K total * (64/32)^2 = 4x
sdpa64 = sdpa32 * 4.0                        # 68 s
nonSDPA64 = 249.0                             # registered 333 - 83.95
total64 = sdpa64 + nonSDPA64

# --- FA flash speedups (full-prefill, per L) -----------------------------
speed32 = 64.756 / 18.441    # 3.51x
speed64 = 143.231 / 39.766   # 3.60x

# FA per-chunk last-chunk ms (Q, prefix): to get pc=8192 flash SDPA
#   32K: Q=2048 last=37.767ms; Q=8192 last=114.533ms
# full prefill (n chunks growing): sum = last * n(n+1)/2 / n = last*(n+1)/2
def full_prefill(last_ms, n_chunks):
    return last_ms * (n_chunks + 1) / 2 / 1000.0   # s per layer
N_FULL_LAYERS = 16
# FA per-chunk LAST-chunk time in ms (per full-attn layer), from the FA bench:
#   32K Q=2048 last=37.767ms, Q=8192 last=114.533ms
#   64K Q=2048 last=81.44ms (39.766us/tok*2048), Q=8192 last=270.91ms (33.070us/tok*8192)
flash_sdpa = {
    (32, 2048): full_prefill(37.767, 32780 // 2048) * N_FULL_LAYERS,   # 4.87 s
    (32, 8192): full_prefill(114.533, 32780 // 8192) * N_FULL_LAYERS,  # 4.58 s
    (64, 2048): full_prefill(81.440, 65548 // 2048) * N_FULL_LAYERS,   # 9.5 s
    (64, 8192): full_prefill(270.911, 65548 // 8192) * N_FULL_LAYERS,  # 9.0 s
}

# dense SDPA at each (L, pc) ~ O(L^2) independent of pc (same total)
dense_sdpa = {(32, 2048): sdpa32, (32, 8192): sdpa32, (64, 2048): sdpa64, (64, 8192): sdpa64}

# FFN per-token rises with pc (MCP1 M-curve, v0.32.2): 8192/2048 = 8.321/7.839
mc_ratio = 8.321 / 7.839
ffn = {2048: ffn32, 8192: ffn32 * mc_ratio}

def total_cell(L, pc, use_flash):
    base_total = total32 if L == 32 else total64
    sdpa = flash_sdpa[(L, pc)] if use_flash else dense_sdpa[(L, pc)]
    ffn_term = ffn[pc] if L == 32 else ffn[2048] * (L / 32780)   # 64K FFN ~ O(L)
    nonSDPA_base = (base_total - dense_sdpa[(L, 2048)])
    # non-SDPA = FFn(pc) + (GDN+other)  ; GDN+other flat vs pc
    gdn_other = nonSDPA_base - ffn[2048]
    return ffn_term + gdn_other + sdpa

print("=== FB0 predicted end-to-end prefill wall (s) ===")
print(f"{'cell':24s} {'32K':>8s} {'64K':>8s}")
for pc in (2048, 8192):
    for flash in (False, True):
        t32 = total_cell(32, pc, flash)
        t64 = total_cell(64, pc, flash)
        label = f"{'flash' if flash else 'dense'} pc={pc}"
        print(f"{label:24s} {t32:8.1f} {t64:8.1f}")

print("\n=== KEEP gate: % saving of total vs (dense, pc=2048) baseline ===")
base32 = total_cell(32, 2048, False)
base64 = total_cell(64, 2048, False)
print(f"baseline (dense,pc=2048): 32K={base32:.1f}s  64K={base64:.1f}s")
for pc in (2048, 8192):
    s32 = 100 * (1 - total_cell(32, pc, True) / base32)
    s64 = 100 * (1 - total_cell(64, pc, True) / base64)
    print(f"flash pc={pc}:  32K {s32:+.1f}%  64K {s64:+.1f}%   (32K gate >=10%: {'PASS' if s32>=10 else 'FAIL'})")
