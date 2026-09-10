// StopSequenceMatcherTests.swift
//
// Unit tests for the incremental stop-sequence matcher of the Qwen 3.8 MTP
// server.
//
// The matcher is pure Swift (no MLX, no model weights), so these tests run on
// any machine without loading the model. They pin the incremental matching
// behavior: emit-safe prefixes, stop detection within and across fragment
// boundaries, multi-byte (grapheme) handling, earliest-match selection across
// multiple stop sequences, and post-stop suppression.

import Testing
import Foundation
@testable import HTTPServer

// MARK: - Empty / no-match

@Test
func noStopSequencesEmitsAll() {
    var matcher = StopSequenceMatcher(sequences: [])
    #expect(matcher.hasStopSequences == false)
    #expect(matcher.consume("hello") == "hello")
    #expect(matcher.isStopped == false)
}

@Test
func noMatchEmitsWholeFragment() {
    var matcher = StopSequenceMatcher(sequences: ["STOP"])
    let emitted = matcher.consume("hello world")
    #expect(emitted == "hello world")
    #expect(matcher.isStopped == false)
}

// MARK: - Stop within a single fragment

@Test
func stopWithinSingleFragment() {
    var matcher = StopSequenceMatcher(sequences: ["STOP"])
    let emitted = matcher.consume("hello STOP world")
    #expect(emitted == "hello ")
    #expect(matcher.isStopped == true)
    #expect(matcher.consume("more") == "")
}

@Test
func stopAtStartOfFirstFragment() {
    var matcher = StopSequenceMatcher(sequences: ["STOP"])
    let emitted = matcher.consume("STOP hello")
    #expect(emitted == "")
    #expect(matcher.isStopped == true)
}

// MARK: - Stop spanning a fragment boundary

@Test
func stopSpansFragmentBoundary() {
    var matcher = StopSequenceMatcher(sequences: ["STOP"])
    let first = matcher.consume("hello STO")
    #expect(first == "hello STO")
    #expect(matcher.isStopped == false)
    let second = matcher.consume("P world")
    #expect(second == "")
    #expect(matcher.isStopped == true)
}

@Test
func stopSequencePrefixAlreadyEmitted() {
    // The "STO" prefix is emitted in the first fragment; only the completing
    // "P" is suppressed. This is the documented incremental behavior: a prefix
    // of a stop sequence emitted before the sequence completes stays in the
    // output.
    var matcher = StopSequenceMatcher(sequences: ["STOP"])
    let first = matcher.consume("hello STO")
    #expect(first == "hello STO")
    let second = matcher.consume("P")
    #expect(second == "")
    #expect(matcher.isStopped == true)
}

// MARK: - Multi-byte (grapheme) handling

@Test
func multiByteStopSpanningFragmentBoundary() {
    // "café" is four Characters; the multi-byte "é" completes the stop
    // sequence in the second fragment. The matcher works in Character space,
    // so the grapheme is never split across fragments and the carry retains it
    // whole.
    var matcher = StopSequenceMatcher(sequences: ["café"])
    let first = matcher.consume("hello caf")
    #expect(first == "hello caf")
    #expect(matcher.isStopped == false)
    let second = matcher.consume("é world")
    #expect(second == "")
    #expect(matcher.isStopped == true)
}

@Test
func multiByteCharacterIsPreservedWhenNotAStop() {
    var matcher = StopSequenceMatcher(sequences: ["STOP"])
    let emitted = matcher.consume("café 🚀")
    #expect(emitted == "café 🚀")
    #expect(matcher.isStopped == false)
}

// MARK: - Multiple stop sequences

@Test
func multipleStopSequencesEarliestMatch() {
    var matcher = StopSequenceMatcher(sequences: ["STOP", "END"])
    let emitted = matcher.consume("hello END STOP")
    #expect(emitted == "hello ")
    #expect(matcher.isStopped == true)
}

// MARK: - Post-stop

@Test
func afterStopSuppressesAll() {
    var matcher = StopSequenceMatcher(sequences: ["STOP"])
    #expect(matcher.consume("STOP") == "")
    #expect(matcher.isStopped == true)
    #expect(matcher.consume("hello") == "")
    #expect(matcher.consume("world") == "")
}

@Test
func emptyFragmentReturnsEmpty() {
    var matcher = StopSequenceMatcher(sequences: ["STOP"])
    #expect(matcher.consume("") == "")
    #expect(matcher.isStopped == false)
}
