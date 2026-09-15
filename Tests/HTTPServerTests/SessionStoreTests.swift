// SessionStoreTests.swift
//
// Unit tests for the process-local, in-memory `SessionStore` actor. These are
// pure actor tests: no model, no weights, no network. They pin the session
// ownership contract: conversation history continuity, in-flight serialization,
// bounded LRU/TTL lifecycle, and the diagnostic cached-prefix release on delete.

import Testing
import Foundation
@testable import HTTPServer

private func makeCommit(
    content: String? = "hi",
    reasoningContent: String? = nil,
    toolCalls: [ToolCall] = [],
    completionTokenIDs: [Int] = [1, 2, 3],
    seedTokens: [Int] = [10, 11],
    promptTokens: Int = 2,
    completionTokens: Int = 3
) -> CompletionCommit {
    CompletionCommit(
        content: content,
        reasoningContent: reasoningContent,
        toolCalls: toolCalls,
        completionTokenIDs: completionTokenIDs,
        seedTokens: seedTokens,
        promptTokens: promptTokens,
        completionTokens: completionTokens
    )
}

@Test func sessionStoreCreateExistsInspectDelete() async {
    let store = SessionStore(maxSessions: 4, defaultTTL: 60)
    let session = await store.create()
    #expect(!session.id.isEmpty)
    #expect(session.token_count == 0)
    #expect(await store.exists(session.id))
    #expect(await store.session(session.id)?.id == session.id)

    let freed = await store.delete(session.id)
    #expect(freed == nil)  // no completion committed, so no cached prefix
    #expect(await store.exists(session.id) == false)
}

@Test func sessionStoreCompletionLifecycle() async {
    let store = SessionStore(maxSessions: 4, defaultTTL: 60)
    let id = await store.create().id

    // Empty history before any completion.
    let m0 = await store.messages(id)
    #expect(m0?.isEmpty == true)

    // begin/finish serialize per-session; a second begin is rejected while busy.
    #expect(await store.beginCompletion(id))
    #expect(await store.beginCompletion(id) == false)

    // Commit a user turn plus the assistant reply.
    await store.finishCompletion(
        id,
        requestMessages: [ChatMessage(role: "user", content: "hi")],
        commit: makeCommit()
    )

    let msgs = await store.messages(id)
    #expect(msgs?.count == 2)
    #expect(msgs?[0].role == "user")
    #expect(msgs?[0].content == "hi")
    #expect(msgs?[1].role == "assistant")
    #expect(msgs?[1].content == "hi")  // commit content carried through

    // token count is prompt + completion (2 + 3).
    #expect(await store.session(id)?.token_count == 5)

    // After finish the session is free for the next completion.
    #expect(await store.beginCompletion(id))
    await store.abortCompletion(id)
}

@Test func sessionStoreDeleteReleasesCachedPrefix() async {
    let store = SessionStore(maxSessions: 4, defaultTTL: 60)
    let id = await store.create().id
    await store.beginCompletion(id)
    await store.finishCompletion(
        id,
        requestMessages: [ChatMessage(role: "user", content: "hi")],
        commit: makeCommit()
    )
    // Cached prefix = seed (2) + completion (3) = 5 tokens.
    let freed = await store.delete(id)
    #expect(freed?.count == 5)
}

@Test func sessionStoreAbortedCompletionCommitsNothing() async {
    let store = SessionStore(maxSessions: 4, defaultTTL: 60)
    let id = await store.create().id
    #expect(await store.beginCompletion(id))
    await store.abortCompletion(id)
    // No messages committed; no cached prefix.
    let mEmpty = await store.messages(id)
    #expect(mEmpty?.isEmpty == true)
    let freed = await store.delete(id)
    #expect(freed == nil)
}

@Test func sessionStoreLRUEvictsWhenOverCap() async {
    let store = SessionStore(maxSessions: 2, defaultTTL: 60)
    let a = await store.create()
    let b = await store.create()
    let c = await store.create()
    // Oldest (a) is evicted to stay under the cap.
    #expect(await store.exists(a.id) == false)
    #expect(await store.exists(b.id))
    #expect(await store.exists(c.id))
}

@Test func sessionStoreExpiresAfterTTL() async {
    let store = SessionStore(maxSessions: 100, defaultTTL: 0.001)  // 1 ms
    let id = await store.create().id
    #expect(await store.exists(id))
    try? await Task.sleep(for: .milliseconds(10))
    #expect(await store.exists(id) == false)
}
