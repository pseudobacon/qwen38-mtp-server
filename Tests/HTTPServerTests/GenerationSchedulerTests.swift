import XCTest
@testable import HTTPServer

/// A minimal `GenerationProvider` for scheduler tests. It is an actor so
/// that `cancelCalls` is actor-isolated and no manual locking is needed.
actor MockGenerationProvider: GenerationProvider {
    var cancelCalls: [(GenerationID, GenerationCancellationReason)] = []

    func generateStream(id: GenerationID, request: ChatCompletionRequest, samplingParams: SamplingParameters) -> AsyncStream<GenerationFragment> {
        return AsyncStream { $0.finish() }
    }

    func cancelGeneration(id: GenerationID, reason: GenerationCancellationReason) {
        cancelCalls.append((id, reason))
    }
}

final class GenerationSchedulerTests: XCTestCase {
    
    func mockRequest() -> (ChatCompletionRequest, SamplingParameters) {
        let req = ChatCompletionRequest(model: "qwen", messages: [])
        let params = SamplingParameters(temperature: 0.0, topP: 1.0, topK: 1, minP: 0.0, repetitionPenalty: 1.0, presencePenalty: 0.0, frequencyPenalty: 0.0, maxTokens: 10, contextWindow: 100, enableThinking: true, mtpEnabled: true, prefillChunkSize: 512, stopSequences: [], kvCacheConfig: ResolvedKVCacheConfig.default, ttlSeconds: nil)
        return (req, params)
    }
    
    func testFIFOAdmissionAndSingleActiveJob() async throws {
        let scheduler = GenerationScheduler(provider: MockGenerationProvider(), maxQueueSize: 5)
        let (req, params) = mockRequest()
        
        // Schedule three requests
        let id1 = GenerationID(), id2 = GenerationID(), id3 = GenerationID()
        
        // Admit 1 first (the lane is free, so this returns immediately).
        // Awaiting admission before spawning the next request makes the
        // enqueue order deterministic: `schedule` enqueues from a detached
        // task, so three concurrently spawned schedules have no defined
        // queue order, and asserting a specific admission order over them
        // is a race (it deadlocks when a later ID enqueues first, because
        // `releaseLane` for the wrong ID is a no-op).
        _ = try await scheduler.schedule(id: id1, request: req, samplingParams: params)
        let active1 = await scheduler.activeId
        let depth1 = await scheduler.queueDepth
        XCTAssertEqual(active1, id1)
        XCTAssertEqual(depth1, 0)
        
        // Enqueue 2, and wait until it is actually in the queue...
        async let s2 = scheduler.schedule(id: id2, request: req, samplingParams: params)
        while await scheduler.queueDepth < 1 {
            try await Task.sleep(for: .milliseconds(1))
        }
        
        // ...before enqueuing 3, so the FIFO order is id2 then id3.
        async let s3 = scheduler.schedule(id: id3, request: req, samplingParams: params)
        while await scheduler.queueDepth < 2 {
            try await Task.sleep(for: .milliseconds(1))
        }
        let depth = await scheduler.queueDepth
        XCTAssertEqual(depth, 2)
        
        // Release 1, admitting 2
        await scheduler.releaseLane(id: id1)
        _ = try await s2
        let active2 = await scheduler.activeId
        XCTAssertEqual(active2, id2)
        
        // Release 2, admitting 3
        await scheduler.releaseLane(id: id2)
        _ = try await s3
        let active3 = await scheduler.activeId
        XCTAssertEqual(active3, id3)
    }
    
    func testOverloadRejection() async throws {
        let scheduler = GenerationScheduler(provider: MockGenerationProvider(), maxQueueSize: 1)
        let (req, params) = mockRequest()
        
        // Fill active lane
        async let s1 = scheduler.schedule(id: GenerationID(), request: req, samplingParams: params)
        _ = try await s1 
        
        // Fill queue (limit 1)
        Task { try await scheduler.schedule(id: GenerationID(), request: req, samplingParams: params) }
        
        // Allow enqueuing tasks a moment to run
        try await Task.sleep(for: .milliseconds(50)) 
        
        // Third should reject immediately
        do {
            _ = try await scheduler.schedule(id: GenerationID(), request: req, samplingParams: params)
            XCTFail("Should have thrown queueFull")
        } catch GenerationScheduler.SchedulerError.queueFull {
            XCTAssertTrue(true)
        }
    }
    
    func testQueuedCancellation() async throws {
        let scheduler = GenerationScheduler(provider: MockGenerationProvider(), maxQueueSize: 5)
        let (req, params) = mockRequest()
        let id1 = GenerationID(), id2 = GenerationID()
        
        // Fill active lane
        async let _ = scheduler.schedule(id: id1, request: req, samplingParams: params)
        try await Task.sleep(for: .milliseconds(50))
        
        // Queue second request
        let task2 = Task { try await scheduler.schedule(id: id2, request: req, samplingParams: params) }
        try await Task.sleep(for: .milliseconds(50))
        
        // Cancel the queued request
        task2.cancel()
        
        do {
            _ = try await task2.value
            XCTFail("Should have thrown cancellation")
        } catch GenerationScheduler.SchedulerError.cancelledBeforeAdmission {
            let totalCancelled = await scheduler.totalCancelled
            XCTAssertEqual(totalCancelled, 1)
        }
    }
    
    func testActiveCancellationForwardsToProvider() async throws {
        let provider = MockGenerationProvider()
        let scheduler = GenerationScheduler(provider: provider, maxQueueSize: 5)
        let (req, params) = mockRequest()
        let id = GenerationID()
        
        _ = try await scheduler.schedule(id: id, request: req, samplingParams: params)
        await scheduler.cancel(id: id, reason: .callerCancelled)
        
        // The active ID remains until explicitly released, but the provider is signalled
        let activeId = await scheduler.activeId
        XCTAssertEqual(activeId, id)
        
        let providerCalls = await provider.cancelCalls
        XCTAssertEqual(providerCalls.first?.0, id)
    }
}