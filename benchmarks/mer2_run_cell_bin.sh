#!/bin/bash
# MER2 cell wrapper: place a specific binary+metallib into .build/release/
# (where run_cell.sh and the runtime search path expect them), then delegate to
# run_cell.sh. The binary's colocated metallib is placed alongside it so the
# first runtime search path (<exe-dir>/mlx.metallib) resolves.
#
# usage: mer2_run_cell_bin.sh <cell> <bin-dir> <tag> <env-spec> <port> <prompt-file> [expect-head] [max-tokens]
#   bin-dir: directory containing qwen38-mtp-server + mlx.metallib
set -u
CELL=$1
BINDIR=$2
TAG=$3
ENVSPEC=$4
PORT=${5:-18099}
PROMPT_FILE=$6
EXPECT_HEAD=${7:-q4}
MAX_TOKENS=${8:-1024}

SERVER=/Users/cwong/ai/qwen38-mtp-server
RELEASE=$SERVER/.build/release
DIR=$SERVER/benchmarks

BIN_SRC="$BINDIR/qwen38-mtp-server"
METAL_SRC="$BINDIR/mlx.metallib"
[ -x "$BIN_SRC" ] || { echo "{\"error\":\"missing binary $BIN_SRC\"}"; exit 1; }
[ -f "$METAL_SRC" ] || { echo "{\"error\":\"missing metallib $METAL_SRC\"}"; exit 1; }

# swap in the target binary + colocated metallib
cp -f "$BIN_SRC" "$RELEASE/qwen38-mtp-server"
cp -f "$METAL_SRC" "$RELEASE/mlx.metallib"
# Re-establish a valid ad-hoc signature after the copy (AMFI/Gatekeeper kills an
# unsigned/ad-hoc binary copied to a new path: "Unrecoverable CT signature issue").
codesign --force --sign - "$RELEASE/qwen38-mtp-server" 2>/dev/null || true

# delegate (run_cell.sh records the swapped-in binary's SHA-256 + HEADs)
"$DIR/run_cell.sh" "$TAG" "$ENVSPEC" "$PORT" "$PROMPT_FILE" "" "$EXPECT_HEAD" "$MAX_TOKENS"
