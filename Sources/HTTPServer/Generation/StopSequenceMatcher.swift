// StopSequenceMatcher.swift
//
// Incremental matching of user-provided stop sequences against committed
// decoded output text.
//
// Pure Swift (no MLX, no model weights). The matcher is fed the emitted text
// fragments in order and reports, for each fragment, the portion that is safe
// to emit (up to but not including a completed stop sequence) and whether a
// stop sequence has been completed.
//
// It handles stop sequences that span decoded fragments by retaining a carry
// of the tail of the previously emitted text, bounded by the longest stop
// sequence minus one character. All matching is done in Swift `String`
// (Character) space, so multi-byte UTF-8 sequences are never split and grapheme
// clusters are preserved.

import Foundation

/// Incrementally matches user-provided stop sequences against committed
/// decoded text, handling sequences that span decoded fragments.
///
/// The matcher is fed emitted text fragments in stream order. For each
/// fragment it returns the emit-safe prefix (the portion of the fragment that
/// precedes any completed stop sequence) and flags whether a stop sequence has
/// been completed. Once stopped, all subsequent fragments are suppressed.
struct StopSequenceMatcher: Sendable {
    private let sequences: [String]
    private let maxSequenceLength: Int
    /// Tail of the cumulative emitted stream, bounded to `maxSequenceLength - 1`
    /// characters, used to detect stop sequences that span fragment boundaries.
    private var carry: String = ""
    private var stopped = false

    /// Creates a matcher for the given stop sequences.
    /// - Parameter sequences: stop sequences (already validated to be
    ///   non-empty and within the documented limits).
    init(sequences: [String]) {
        self.sequences = sequences
        self.maxSequenceLength = sequences.map(\.count).max() ?? 0
    }

    /// Whether any stop sequence has been completed.
    var isStopped: Bool { stopped }

    /// Whether the matcher has any stop sequences to match.
    var hasStopSequences: Bool { !sequences.isEmpty }

    /// Feeds a fragment of committed text and returns the portion that is safe
    /// to emit (up to but not including a completed stop sequence).
    ///
    /// - Returns: the emit-safe prefix of `text`. Empty when a stop sequence
    ///   has completed with no new text preceding it, when the matcher is
    ///   already stopped, or when `text` is empty.
    ///
    /// Note: because matching is incremental, a prefix of a stop sequence that
    /// was already emitted in an earlier fragment (before the sequence
    /// completed) remains in the output. The matcher guarantees that no text
    /// *past* a completed stop sequence is emitted.
    mutating func consume(_ text: String) -> String {
        guard !stopped, !text.isEmpty else {
            return ""
        }

        // No stop sequences configured: all text is safe to emit.
        guard !sequences.isEmpty else {
            return text
        }

        let candidate = carry + text

        // Find the earliest occurrence of any stop sequence.
        var matchIndex: Int?
        for sequence in sequences {
            if let range = candidate.range(of: sequence) {
                let index = candidate.distance(
                    from: candidate.startIndex,
                    to: range.lowerBound
                )
                if matchIndex == nil || index < matchIndex! {
                    matchIndex = index
                }
            }
        }

        guard let index = matchIndex else {
            // No stop sequence completed: emit the whole fragment and extend
            // the carry to the tail of the cumulative emitted stream.
            carry = Self.tail(carry + text, length: maxSequenceLength - 1)
            return text
        }

        // A stop sequence completed. Emit only the portion of `text` that
        // precedes the match; the carry portion was already emitted in an
        // earlier fragment.
        let emitCount = max(0, index - carry.count)
        stopped = true
        carry = ""
        return String(text.prefix(emitCount))
    }

    /// The last `length` characters of `text`, or the whole text if it is
    /// shorter. Never splits a grapheme cluster.
    private static func tail(_ text: String, length: Int) -> String {
        guard length > 0 else { return "" }
        if text.count <= length {
            return text
        }
        let index = text.index(text.endIndex, offsetBy: -length)
        return String(text[index...])
    }
}
