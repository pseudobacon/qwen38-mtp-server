#!/usr/bin/env python3
"""LEV-D: Flash + large-pc compounding model.
total(pc) = FFN(pc) + GDN(pc) + SDPA(pc) + other
Solve for c_flash (per-token SDPA cost of a flash kernel) at which
pc>=8192 (enabled by flash removing the scores buffer) beats the
pc=2048 incumbent (dense SDPA) at 32K and 64K.
All inputs from registered in-session data:
  - MCP2 32K per-phase (pc 512/1024/2048), single session  mcp-20260917
  - MCP1/MER4A.2 FFN M-curve (per-layer per-token, through M=8192)
  - prefill-verify-2026-09-15 64K pc=512 per-phase (SDPA share 29.3%)
"""
L32, L64 = 32768, 65536

# MCP2 32K per-phase, ms (single session, pc=512/1024/2048)
mcp2_32k = {
    512:  dict(FFN=72408, GDN=35737, sdpa=27574, attn=37176, norm=2128, resid=2039, total=149488),
    1024: dict(FFN=67265, GDN=34614, sdpa=26262, attn=34863, norm=1363, resid=1333, total=139438),
    2048: dict(FFN=65259, GDN=30998, sdpa=24704, attn=32416, norm=978,  resid=925,  total=130575),
}
def other_ms(d): return (d["attn"] - d["sdpa"]) + d["norm"] + d["resid"]  # QKV+O+RoPE+norm+resid

# per-token (us/tok) at 32K
def ptok(ms, L): return ms*1e3/L
print("=== 32K per-token (us/tok), MCP2 single session ===")
for pc in (512,1024,2048):
    d = mcp2_32k[pc]
    print(f"pc={pc:5d}  FFN {ptok(d['FFN'],L32):7.1f}  GDN {ptok(d['GDN'],L32):7.1f}  "
          f"SDPA {ptok(d['sdpa'],L32):7.1f}  other {ptok(other_ms(d),L32):7.1f}  "
          f"total {ptok(d['total'],L32):8.1f} us/tok")

# --- extrapolate FFN & GDN to pc=8192 -------------------------------
# (a) A + B/pc fit on the 3 end-to-end points (per-chunk overhead drops with pc)
def fit_abpc(vals):  # vals: dict pc->per-token
    (p1,v1),(p2,v2) = sorted(vals.items())[:2]
    B = (v1-v2)*p2            # v1-v2 = B/p1 - B/p2 = B/p2  (p2=2*p1)
    A = v1 - B/p1
    return A, B
ffn_pt = {pc: ptok(mcp2_32k[pc]["FFN"],L32) for pc in (512,1024,2048)}
gdn_pt = {pc: ptok(mcp2_32k[pc]["GDN"],L32) for pc in (512,1024,2048)}
A_f,B_f = fit_abpc(ffn_pt); A_g,B_g = fit_abpc(gdn_pt)
ffn_8192_fit = A_f + B_f/8192
gdn_8192_fit = A_g + B_g/8192

# (b) M-curve ratio (isolated GEMM per-layer per-token): 8192/2048
mc_v311 = {512:11.261,1024:6.426,2048:7.528,4096:8.190,8192:7.966}
mc_v322 = {512:13.258,1024:7.301,2048:7.839,4096:7.932,8192:8.321}
r311 = mc_v311[8192]/mc_v311[2048]; r322 = mc_v322[8192]/mc_v322[2048]
ffn_8192_mc = ffn_pt[2048]*r311
print("\n=== FFN/GDN @ pc=8192 (us/tok) ===")
print(f"FFN  A+B/pc fit      : {ffn_8192_fit:8.1f}  (vs 2048 {ffn_pt[2048]:.1f}, {100*(ffn_8192_fit/ffn_pt[2048]-1):+5.1f}%)")
print(f"FFN  M-curve x{r311:.3f}   : {ffn_8192_mc:8.1f}  (vs 2048, {100*(r311-1):+5.1f}%)  [v0.31.1]")
print(f"FFN  M-curve x{r322:.3f}   : {ffn_pt[2048]*r322:8.1f}  (v0.32.2)")
print(f"GDN  A+B/pc fit      : {gdn_8192_fit:8.1f}  (vs 2048 {gdn_pt[2048]:.1f})")

# bound FFN(8192): optimistic = fit (per-chunk overhead wins), pessimistic = M-curve (GEMM rises)
ffn81_lo, ffn81_hi = min(ffn_8192_fit, ffn_pt[2048]), max(ffn_8192_fit, ffn_pt[2048]*r322)
gdn81  = gdn_pt[2048]                      # O(L), ~flat at 8192 (per-chunk overhead drops)
oth81  = other_ms(mcp2_32k[2048])*1e3/L32  # ~flat

def bar(L, c_dense, ffn81, gdn81, oth81, ffn2048, gdn2048, oth2048):
    # non-SDPA delta (8192 vs 2048), us/tok
    d = (ffn81+gdn81+oth81) - (ffn2048+gdn2048+oth2048)
    return c_dense - d, d

print("\n=== GO bar: c_flash (us/tok) where pc=8192+flash beats pc=2048+dense ===")
# 32K
c_dense32 = ptok(mcp2_32k[2048]["sdpa"], L32)
for label, f81 in (("FFN=fit(8192 better)", ffn_8192_fit),
                   ("FFN=M-curve(8192 worse)", ffn_pt[2048]*r322)):
    b, d = bar(L32, c_dense32, f81, gdn81, oth81,
               ffn_pt[2048], gdn_pt[2048], oth81)
    print(f"32K {label:24s}: c_flash < {b:7.1f} us/tok   (non-SDPA delta {d:+7.1f} us/tok)")
# 64K: c_dense from O(L^2) scaling of 32K and the 64K pc=512 measurement
c_dense64_l2 = c_dense32 * 2.0            # O(L^2) per-token ~ 2x
c_dense64_meas = 83.95e6/L64 * 0.8959      # 64K pc=512 SDPA (83.95 s, prefill-verify) x 32K pc-ratio(2048/512)
c_dense64 = (c_dense64_l2 + c_dense64_meas)/2
print(f"64K c_dense(2048) estimate: O(L^2) 2x={c_dense64_l2:.0f}, measured-x-ratio={c_dense64_meas:.0f} -> use {c_dense64:.0f} us/tok")
for label, f81 in (("FFN=fit(8192 better)", ffn_8192_fit),
                   ("FFN=M-curve(8192 worse)", ffn_pt[2048]*r322)):
    b, d = bar(L64, c_dense64, f81, gdn81, oth81,
               ffn_pt[2048], gdn_pt[2048], oth81)
    print(f"64K {label:24s}: c_flash < {b:7.1f} us/tok   (non-SDPA delta {d:+7.1f} us/tok)")

print("\n=== credibility of the bar vs a realistic Metal flash kernel ===")
# FLOP floor + KV-read floor
flop32 = 13.4e12/100e12*1e3            # ~130 ms at 32K
def kv_bytes(L, pc):
    n = L//pc
    return pc * n*(n-1)//2 * 64*1024     # sum over blocks of (i*pc K-tokens) x 64KiB
bw = 200e9
for pc in (8192, 2048):
    kb = kv_bytes(L32, pc)
    print(f"32K flash KV-read floor pc{pc:5d}: {kb/bw*1e3:7.1f} ms  = {kb/bw*1e6/L32:6.2f} us/tok  ({kb/2**30:5.1f} GiB read)")
# a credible kernel: within ~5-50x FLOP/KV floor
lo, hi = 5*flop32*1e3/L32, 50*flop32*1e3/L32
print(f"credible flash c_flash (5-50x floor): {lo:.0f}-{hi:.0f} us/tok @32K")
