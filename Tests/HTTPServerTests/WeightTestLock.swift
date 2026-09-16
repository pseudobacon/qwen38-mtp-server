// WeightTestLock.swift
//
// Process-level guard for weight-gated tests. Swift Testing runs top-level
// tests concurrently by default; two weight-gated tests each loading the
// ~14 GB model (2 x ~32 GB footprint) on the 48 GB machine exhausts unified
// memory, and a model-load thread that dies mid-load leaves MLX's global
// eval lock permanently held, deadlocking every other test in the process.
//
// Each weight-gated test must call `acquireWeightTestLock()` first and
// return (skipping) when another one already holds it. Run weight-gated
// tests in SEPARATE `swift test` invocations:
//
//   QWEN_RUN_WEIGHT_TESTS=1 swift test --filter mtpCorrectnessSerialMatchesMTPAtAllDepths
//   QWEN_RUN_WEIGHT_TESTS=1 swift test --filter serialAndMTPMatchDistributionally

import Darwin
import Foundation

// Single mutable fd, guarded by `nonisolated(unsafe)`: the lock file is only
// ever opened once per process and the value is never read/written from
// concurrent paths in a racy way (first caller wins, later callers observe a
// non-negative fd and return). `@unchecked`-free because it is a plain Int32.
// nonisolated(unsafe) is the minimal, documented escape hatch for this
// process-lifetime scalar.
private nonisolated(unsafe) var weightTestLockFD: Int32 = -1

/// Acquire the process-wide weight-test lock. Returns `false` (and skips
/// nothing — the caller must skip) when another test in this process already
/// holds it. The fd stays open for the process lifetime, so the lock is
/// released automatically at exit.
func acquireWeightTestLock() -> Bool {
    if weightTestLockFD >= 0 { return true }
    let path = "/tmp/qwen-weight-gated-test.lock"
    let fd = open(path, O_RDWR | O_CREAT, 0o644)
    guard fd >= 0 else { return true }  // cannot create: proceed unlocked
    guard flock(fd, LOCK_EX | LOCK_NB) == 0 else {
        close(fd)
        return false
    }
    weightTestLockFD = fd
    return true
}
