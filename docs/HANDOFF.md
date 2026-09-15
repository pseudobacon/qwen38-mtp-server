# HANDOFF — Fused GDN prework kernel (MLX_QWEN_FUSED_GDN) (IN PROGRESS — engine merged, server docs pending)

> **Checkpoint status.** The fresh-checkpoint procedure **must** be run via
> `./scripts/agent-checkpoint.sh` before this file is finalized (it writes
> `.dsh/last-agent-checkpoint` in each repo). Markers in the Checkpoint
> markers section below.

## Objective and acceptance criteria

Port `qwen35PackedGDNPreworkKernel` (from the oMLX/mlx-serve reference) into
the engine so that the GDN prologue (conv1d + SiLU + Q/K/V split + Q/K
rmsNorm-and-scale + g/beta) runs as **one Metal launch** for MTP verify widths
S ∈ 3…9. Acceptance: kernel transcribed **verbatim** for the
bit-exact-critical logic (0xC0DB→0x3A8B beta fixup, `precise::exp`, barriers,
stride access); gated **OFF by default** (`MLX_QWEN_FUSED_GDN=1` to enable);
**bit-exact** with the eager chain (proven, not assumed); engagement provable
(dispatch counter > 0, not a silent gate miss); the non-fused path is
bit-identical when the gate is off (zero overhead); tests; performance
benchmark (honest); docs; engine commit precedes server commit.

## Result

**Fused GDN prework (engine merged to main at `4cd8603`; server docs this
commit).** No sampling/greedy semantics change — the kernel is bit-exact with
the eager chain, so committed tokens are unchanged. Off by default.

- **Engine** (`mlx-swift-lm`, `4cd8603`):
  - `Qwen35Kernels.swift`: `qwen35PackedGDNPreworkKernel` (verbatim Metal
    source + header, `ensureRowContiguous: false`), `Qwen35FusedGDNPreworkRouting`
    (env flag, off by default), `Qwen35FusedGDNPreworkDispatch` (lock-protected
    per-width + total counters), `qwen35PackedGDNPrework` helper.
  - `Qwen35+FastPath.swift`: `fusedGDNPrework(qkv:a:b:convState:)` on
    `Qwen35GatedDeltaNet` — the full geometry/dtype gate + the kernel call.
  - `Qwen35.swift`: `forward` uses `if mask == nil, let prework =
    fusedGDNPrework(...)` with the **unchanged** eager chain in the else branch;
    per-instance `fusedGDNPreworkEnabled` (default from the env flag) for
    testability; a load-time fusion summary line.
  - `Qwen35FusedGDNProjectionTests`: `testFusedGDNPreworkIsBitIdenticalAcrossVerifyWidths`
    (S ∈ 3…9 bit-exact + engagement), `testFusedGDNPreworkWidthGate`
    (S ∈ {1,2,10,16} gated off). 17/17 green.

### Metal-version adaptation (minimal, bit-exact)

The verbatim kernel did not compile under this MLX's Metal, which promotes
`bfloat16_t op bfloat16_t` to `float` (the reference's Metal keeps it in
`bfloat16_t`). Three `const InT x = <InT op InT>` lines needed explicit
`static_cast<InT>(static_cast<float>(a) op static_cast<float>(b))`. Bit-exact
(Metal emulates bf16 in float32). The 0xC0DB→0x3A8B beta fixup, `precise::exp`,
barriers, and stride access are **untouched**.

### Bit-exactness (proven)

- **Unit** (no 14 GB weights): full GDN layer output bit-identical (fused vs
  eager) for S ∈ 3…9 on the production geometry; dispatch counter proves
  engagement. The unit test casts in_proj / conv1d.weight / A_log / dt_bias to
  bf16 to match production's dequantized dtypes (a fresh float32 layer would
  emit float32 and the gate would miss).
- **Real model** (14 GB 27B, `Qwen38MTPDiagnosticTests`): `MLX_QWEN_FUSED_GDN=1`
  → overall committed stream hash
  `86cd9e8868988aef611512262936a11d717bf8f6882efa8997910b992784b1c5` (identical
  to the OFF baseline) and `[FusedGDNDispatch] engagedTotal=9216 widths=[5: 9216]`
  — engaged on every verify round, not a silent gate miss. Wide-verify (depth 5)
  serial/wide hashes also match the OFF baseline.

### Performance (honest finding)

Controlled in-session A/B of the diagnostic test (3 reps each, median): S=5
(OFF 33.64 s vs ON 36.67 s), S=6 wide-verify (OFF 30.99 s vs ON 31.55 s).
**Bit-exact but not a clear wall-clock win** in these workloads (the verify path
is a small fraction of total wall-clock; fusing small-tensor ops does not beat
the eager chain's already-cheap small launches). Retained as a
correctness-preserving, default-OFF opt-in, not a headline win. No absolute
tok/s cited (cross-session absolutes are not comparable).

## Git state

- `../mlx-swift-lm`: branch `main`, HEAD `4cd8603` (the fused GDN kernel), the
  merge of `feature/prompt-fused-gdn` (branch deleted). Working tree clean.
- `qwen38-mtp-server` (this repo): branch `feature/prompt-fused-gdn` (off
  `main` @ `e994ea5`); this commit carries the `progress.md` + `docs/HANDOFF.md`
  updates.

## Commands / verification

```
cd ../mlx-swift-lm
swift build --target MLXLLM                                  # clean
swift test --filter Qwen35FusedGDNProjectionTests            # 17 green
MLX_QWEN_FUSED_GDN=1 swift test --filter Qwen38MTPDiagnosticTests  # hashes match baseline, engagedTotal=9216
```

## Unresolved risks / caveats

- The fused kernel is **off by default**; enabling it (`MLX_QWEN_FUSED_GDN=1`)
  is bit-exact (proven) but not a measured win in the diagnostic workloads.
- The kernel is gated on the 27B Qwen3.5 geometry; any geometry/dtype/width
  mismatch falls back to the eager chain (silent, by design — a gate miss is
  always safe).
- The Metal-version adaptation (three explicit casts) is the only deviation
  from the verbatim reference; it is bit-exact.

## Do-not-repeat

- Do **not** rewrite the Metal kernel from understanding; transcribe verbatim
  and change only what the Metal version forces (the three explicit casts).
- Do **not** drop `mask == nil` from the `fusedDecode` check in the else branch,
  nor the dtype checks (conv1d.weight/aLog/dtBias must be bf16 for the kernel).
- Do **not** set `ensureRowContiguous: true` (it inserts a full-carrier copy).
- Do **not** cite cross-session absolute tok/s as a conclusion; the fused kernel
  is a wall-clock A/B, not a benchmark cell.
- Do **not** use `try eval(x).asArray(...)` (eval returns Void).
- Engine commit precedes server commit; engine already merged to `main`.

## Next step

Complete the server commit (`progress.md` + this `docs/HANDOFF.md`), merge
`qwen38-mtp-server` `feature/prompt-fused-gdn` to `main`, delete the branch,
then run `./scripts/agent-checkpoint.sh` in both repos and record the fresh
checkpoint markers below.

## Checkpoint markers

- server: (pending — run `./scripts/agent-checkpoint.sh` in `qwen38-mtp-server`)
- engine: (pending — run `./scripts/agent-checkpoint.sh` in `mlx-swift-lm`)
