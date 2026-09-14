#!/usr/bin/env python3
"""Generate the 4-bit group-64 affine MTP draft-head tree (W4).

Reads the pinned BF16 head and writes a new `q4/` tree in the exact layout
the backbone uses for its 4-bit linears (MLX QuantizedLinear, affine):

    <layer>.weight   U32  [out, in/8]
    <layer>.scales   BF16 [out, in/64]
    <layer>.biases   BF16 [out, in/64]

The eight Linear layers (fc, q/k/v/o_proj, gate/up/down_proj) are quantized;
the seven RMSNorm tensors stay BF16, matching the backbone convention (norms
are never quantized in this checkpoint).

The output tree carries bare keys (no prefix) plus a safetensors index, so
`Qwen38MTPHeadAttachment.verifyHeadTree` accepts it exactly like the BF16
tree. The BF16 source tree is never modified.

Usage:
    make_q4_head.py [--src mtp-head/pinned] [--dst mtp-head/q4]

Requires a venv with mlx + safetensors (e.g. /tmp/q4venv).
"""

import argparse
import json
import os
import sys

import mlx.core as mx

# Layers quantized to 4-bit (bare head keys). All in-dim divisible by 64.
LINEAR_KEYS = [
    "fc",
    "layers.0.self_attn.q_proj",
    "layers.0.self_attn.k_proj",
    "layers.0.self_attn.v_proj",
    "layers.0.self_attn.o_proj",
    "layers.0.mlp.gate_proj",
    "layers.0.mlp.up_proj",
    "layers.0.mlp.down_proj",
]

# Norms kept BF16 (bare head keys).
NORM_KEYS = [
    "norm",
    "layers.0.input_layernorm",
    "layers.0.post_attention_layernorm",
    "layers.0.self_attn.q_norm",
    "layers.0.self_attn.k_norm",
    "pre_fc_norm_embedding",
    "pre_fc_norm_hidden",
]

GROUP_SIZE = 64
BITS = 4


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="mtp-head/pinned")
    ap.add_argument("--dst", default="mtp-head/q4")
    args = ap.parse_args()

    src = os.path.abspath(args.src)
    dst = os.path.abspath(args.dst)
    src_tensor = os.path.join(src, "model.safetensors")

    if not os.path.exists(src_tensor):
        sys.exit(f"source tree missing: {src_tensor}")
    if os.path.exists(os.path.join(dst, "model.safetensors")):
        sys.exit(f"destination tree already exists: {dst} (refusing to overwrite)")

    src_cfg = json.load(open(os.path.join(src, "config.json")))

    tensors = mx.load(src_tensor)
    expected = [k + ".weight" for k in LINEAR_KEYS + NORM_KEYS]
    missing = [k for k in expected if k not in tensors]
    if missing:
        sys.exit(f"source tree missing keys: {missing}")

    out: dict = {}
    weight_map: dict = {}
    bytes_in = 0
    bytes_out = 0

    for key in LINEAR_KEYS:
        w = tensors[key + ".weight"]
        nd, ni = w.shape
        if nd % GROUP_SIZE or ni % GROUP_SIZE:
            sys.exit(f"{key}: dims {w.shape} not divisible by group size {GROUP_SIZE}")
        bytes_in += nd * ni * 2
        ql, scales, biases = mx.quantize(w, group_size=GROUP_SIZE, bits=BITS)
        assert ql.dtype == mx.uint32, f"{key}: unexpected quantized dtype {ql.dtype}"
        out[key + ".weight"] = ql
        out[key + ".scales"] = scales
        out[key + ".biases"] = biases
        bytes_out += ql.nbytes + scales.nbytes + biases.nbytes
        for suffix in ("weight", "scales", "biases"):
            weight_map[key + "." + suffix] = "model.safetensors"

    for key in NORM_KEYS:
        n = tensors[key + ".weight"]
        if n.dtype != mx.bfloat16:
            sys.exit(f"{key}: expected BF16, got {n.dtype}")
        out[key + ".weight"] = n
        bytes_out += n.nbytes
        weight_map[key + ".weight"] = "model.safetensors"

    os.makedirs(dst, exist_ok=True)
    dst_tensor = os.path.join(dst, "model.safetensors")
    mx.save_safetensors(dst_tensor, out)

    with open(os.path.join(dst, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=1)
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(src_cfg, f, indent=2)

    print(f"src : {src}")
    print(f"dst : {dst}")
    print(f"linear keys quantized : {len(LINEAR_KEYS)} (4-bit, group {GROUP_SIZE}, affine)")
    print(f"norm keys kept BF16   : {len(NORM_KEYS)}")
    print(f"BF16 bytes in  : {bytes_in / 1e6:.1f} MB")
    print(f"q4 bytes out   : {bytes_out / 1e6:.1f} MB")
    print(f"total tensors  : {len(out)} (index keys: {len(weight_map)})")
    print(f"config copied  : model_type={src_cfg.get('model_type')} (byte-identical source config)")


if __name__ == "__main__":
    main()
