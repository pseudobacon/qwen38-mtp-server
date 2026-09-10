import Foundation

/// Abstracts the generator so we can test the scheduler without model weights.
protocol GenerationProvider: Sendable {
    func generateStream(
        id: GenerationID,
        request: ChatCompletionRequest,
        samplingParams: SamplingParameters
    ) async -> AsyncStream<GenerationFragment>

    func cancelGeneration(
        id: GenerationID,
        reason: GenerationCancellationReason
    ) async
}

actor GenerationScheduler {
    private let provider: any GenerationProvider
    private let maxQueueSize: Int
    
    private struct QueuedRequest {
        let id: GenerationID
        let request: ChatCompletionRequest
        let samplingParams: SamplingParameters
        let continuation: CheckedContinuation<AsyncStream<GenerationFragment>, Error>
        let enqueueTime: ContinuousClock.Instant
    }
    
    private var queue: [QueuedRequest] = []
    private(set) var activeId: GenerationID?
    
    // Requirements #7: Metrics
    var activeRequestCount: Int { activeId != nil ? 1 : 0 }
    var queueDepth: Int { queue.count }
    private(set) var totalCompleted = 0
    private(set) var totalErrors = 0
    private(set) var totalCancelled = 0
    private(set) var cumulativeWaitDuration: Duration = .zero
    
    enum SchedulerError: Error, LocalizedError {
        case queueFull
        case cancelledBeforeAdmission
        
        public var errorDescription: String? {
            switch self {
            case .queueFull: 
                return "The server is currently at capacity. Please try again later."
            case .cancelledBeforeAdmission: 
                return "Request was cancelled before starting."
            }
        }
    }
    
    init(provider: any GenerationProvider, maxQueueSize: Int) {
        self.provider = provider
        self.maxQueueSize = maxQueueSize
    }
    
    /// Schedules a generation request. Suspends without blocking HTTP threads until the lane is free.
    func schedule(
        id: GenerationID,
        request: ChatCompletionRequest,
        samplingParams: SamplingParameters
    ) async throws -> AsyncStream<GenerationFragment> {
        // Requirement #6: Overload rejection
        if queue.count >= maxQueueSize && activeId != nil {
            throw SchedulerError.queueFull
        }
        
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                Task {
                    await self.enqueueAndPump(
                        id: id,
                        request: request,
                        samplingParams: samplingParams,
                        continuation: continuation
                    )
                }
            }
        } onCancel: {
            Task { await self.cancel(id: id, reason: .callerCancelled) }
        }
    }
    
    private func enqueueAndPump(
        id: GenerationID,
        request: ChatCompletionRequest,
        samplingParams: SamplingParameters,
        continuation: CheckedContinuation<AsyncStream<GenerationFragment>, Error>
    ) async {
        let queued = QueuedRequest(
            id: id,
            request: request,
            samplingParams: samplingParams,
            continuation: continuation,
            enqueueTime: .now
        )
        queue.append(queued)
        await pumpQueue()
    }
    
    /// Cancels a request cleanly regardless of its state.
    func cancel(id: GenerationID, reason: GenerationCancellationReason) async {
        // Requirement #4: Cancelled before admission -> never start model work
        if let index = queue.firstIndex(where: { $0.id == id }) {
            let req = queue.remove(at: index)
            req.continuation.resume(throwing: SchedulerError.cancelledBeforeAdmission)
            totalCancelled += 1
            return
        }
        
        // Requirement #5: Cancel active job
        if activeId == id {
            // Forward cancellation to the generator so it hits cooperative boundaries.
            // Note: We DO NOT release the lane here. We wait for the generator to actually halt,
            // the router to finish receiving, and the router to explicitly call `releaseLane`.
            await provider.cancelGeneration(id: id, reason: reason)
        }
    }
    
    /// Waits for the active generation lane to become free, up to `timeout`.
    /// Used during graceful shutdown to let the active generation finish or
    /// hit its cooperative cancellation boundary before the process exits.
    /// Queued (not yet active) requests are left in place; they will be
    /// abandoned when the process exits.
    func drain(timeout: Duration) async {
        let deadline = ContinuousClock.now + timeout
        while activeId != nil {
            if ContinuousClock.now >= deadline {
                break
            }
            try? await Task.sleep(for: .milliseconds(100))
        }
    }

    /// Must be called by the Router `defer` block when the request completes or aborts.
    func releaseLane(id: GenerationID, isError: Bool = false) async {
        guard activeId == id else { return }
        
        activeId = nil
        if isError {
            totalErrors += 1
        } else {
            totalCompleted += 1
        }
        
        await pumpQueue() // Admits the next request in FIFO order
    }
    
    private func pumpQueue() async {
        guard activeId == nil, !queue.isEmpty else { return }
        
        let next = queue.removeFirst()
        activeId = next.id
        cumulativeWaitDuration += next.enqueueTime.duration(to: .now)
        
        // Initiate generation on the provider
        let stream = await provider.generateStream(
            id: next.id,
            request: next.request,
            samplingParams: next.samplingParams
        )
        
        // Unblock the suspended HTTP request with the stream
        next.continuation.resume(returning: stream)
    }
}