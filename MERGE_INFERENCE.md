# Merge Inference Threads

This document records the exploration of folding submission and completion into a single thread.

## Goal

The motivation was straightforward:

- `perf` repeatedly showed that `submission` and `completion` were not both heavily utilized in the
  same way as ingress
- the submission -> batch queue -> completion handoff looked like avoidable structure
- combining them might reduce thread handoff overhead and simplify the GPU-side runtime

The intended end state was:

- one `inference` thread
- same ingress and writer structure
- same request-ring semantics
- same ordered batch completion behavior

## Attempted Design

The attempted design was not "submit one batch and block until it finishes."

Instead, the combined thread owned:

- the submission poller
- the completion poller
- the local submission backlog
- the set of in-flight batches

The loop interleaved three activities:

1. drain visible request-ring events into a local backlog
2. submit new batches while sessions were available and coalescing policy allowed it
3. retire the oldest in-flight batch once its completion became ready

That preserved:

- ordered completion consumption from the disruptor
- multiple in-flight batches, in principle
- the existing writer stage

## What Was Changed

The attempted change introduced a combined `gpu` consumer and rewired server startup so:

- `submission` and `completion` became one `inference` thread
- CPU affinity for `submission` / `completion` was treated as one shared lane
- the old batch queue handoff was bypassed in the runtime path

The implementation was then backed out after validation failed on one of the existing sustain
shapes.

## What Worked

The narrow/deep comparison run looked viable.

Merged run:

- [context.md](/home/sriggin/dev/sean/disrust/artifacts/sustain_capture/20260316_153943/context.md)
- `2` client threads
- `16` connections per client worker
- `window=64`
- `--server-io-threads 4`
- summary:
  - `qps 1009347`
  - `p50 1823.7us`
  - `p95 1887.2us`

Prior split-thread run for comparison:

- [context.md](/home/sriggin/dev/sean/disrust/artifacts/sustain_capture/20260316_132454/context.md)
- same `16x64` client shape
- same `4` ingress-thread server shape
- summary:
  - `qps 979485`
  - `p50 1853.4us`
  - `p95 1933.3us`

So on that workload, the merged design looked slightly better, not worse.

## What Failed

The wider/shallower shape did not complete correctly.

Attempted run:

- [artifacts/sustain_capture/20260316_154116](/home/sriggin/dev/sean/disrust/artifacts/sustain_capture/20260316_154116)
- `2` client threads
- `250` connections per client worker
- `window=2`
- `--server-io-threads 4`

Observed behavior:

- the client printed only the sustain header and never produced interval output
- the server remained alive and burning CPU
- the run had to be killed manually

This was not attributed to a hidden extra `SO_REUSEPORT` server.

What was verified:

- the earlier stray server from a previous port had already been killed before the experiment
- the bad run was on a fresh port (`9910`)
- during the stuck run, the live benchmark processes were the expected pair:
  - one server
  - one client

So the failure was treated as a real regression in the merged design, not an artifact of an errant
listener sharing the port.

## Root Cause And Fix

The original merged implementation had one fatal split-thread assumption left in place:

- when the submission backlog reached `max_batch_slots`, the merged `inference` loop still called
  `reserve_session()`
- `reserve_session()` blocks until a session becomes available

That is safe in the split design because:

- `submission` can block waiting for a session
- while `completion` runs on a different thread and eventually frees that session

It is **not** safe in the merged design because:

- the same merged `inference` thread must both:
  - wait for an available session
  - and retire the in-flight batch that would make that session available

That created a real self-deadlock / liveness failure:

- the request ring filled
- publishing stalled
- the merged `inference` thread sat waiting for a session it was itself responsible for freeing

This was reproduced directly by the sustained pipeline+writer integration test and by the real
`sustain` harness.

The fix was simple and specific:

- the merged `InferenceConsumer` no longer blocks on `reserve_session()`
- it now uses only `try_reserve_session()`
- if no session is available, it returns to the main loop so it can keep retiring completions

In other words:

- split submission may block on session availability
- merged submission/completion may **not**

That was the missing scheduler invariant.

## Updated Result

After fixing the blocking session-reservation path, the merged design is now acceptable for the
tested shapes.

What passes:

- isolated ordering/progress coverage:
  - [tests/submission_completion_integration.rs](/home/sriggin/dev/sean/disrust/tests/submission_completion_integration.rs)
  - split and merged pipeline-only tests both pass
- sustained pipeline+writer integration coverage:
  - split and merged sustained tests both pass
- real sustain harness on the prior failing shapes:
  - narrow/deep:
    - [context.md](/home/sriggin/dev/sean/disrust/artifacts/sustain_capture/20260317_103533/context.md)
    - `2` client threads
    - `16` connections per worker
    - `window=64`
    - completed successfully
    - summary:
      - `qps 962896`
      - `p50 1831.9us`
      - `p95 2095.1us`
  - wide/shallow:
    - [context.md](/home/sriggin/dev/sean/disrust/artifacts/sustain_capture/20260317_103627/context.md)
    - `2` client threads
    - `250` connections per worker
    - `window=2`
    - completed successfully
    - summary:
      - `qps 466734`
      - `p50 1518.6us`
      - `p95 3575.8us`

So the current conclusion is now:

- the merged design is viable
- the original failure was a concrete merged-loop liveness bug
- the bug is fixed by making session reservation non-blocking inside the merged loop

## Remaining Concerns

The merge is working for the shapes exercised so far, but a few concerns remain worth measuring:

- the merged loop should continue to avoid any blocking helper that assumes a separate completion
  thread exists
- if future scheduling changes are made, the sustained pipeline+writer integration test should stay
  in the loop as the first liveness guard
- regression validation should continue to include both:
  - narrow/deep (`16x64`, `2` client workers)
  - wide/shallow (`250x2`, `2` client workers)

## Optimization Log

### Attempt 1: Multi-Action Scheduling Passes

Change:

- [inference.rs](/home/sriggin/dev/sean/disrust/src/pipeline/inference.rs)
- instead of doing at most:
  - one ready completion retirement
  - one batch submission
  per main-loop iteration
- the merged inference lane now:
  - retires up to `8` ready batches per pass
  - submits up to `8` batches per pass

Reasoning:

- the merged lane was still carrying split-era loop granularity
- wide/shallow workloads were especially likely to suffer from per-iteration churn
- if several batches are ready or several sessions are available, there is little value in
  returning to the top of the loop after only one action

Validation:

- [tests/submission_completion_integration.rs](/home/sriggin/dev/sean/disrust/tests/submission_completion_integration.rs)
  still passes in no-CUDA mode
- `cargo clippy --all-targets --no-default-features -- -D warnings` passes

Before / after on the two retained sustain shapes:

1. Narrow/deep (`2` client workers, `16` connections, `window=64`)

Before:

- [context.md](/home/sriggin/dev/sean/disrust/artifacts/sustain_capture/20260317_103533/context.md)
- `qps 962896`
- `p50 1831.9us`
- `p95 2095.1us`
- `p99 3831.8us`

After:

- [context.md](/home/sriggin/dev/sean/disrust/artifacts/sustain_capture/20260317_111652/context.md)
- `qps 983991`
- `p50 1860.6us`
- `p95 1956.9us`
- `p99 2226.2us`

Interpretation:

- throughput improved by about `2.2%`
- p50 regressed slightly
- p95 improved
- p99 improved substantially

2. Wide/shallow (`2` client workers, `250` connections, `window=2`)

Before:

- [context.md](/home/sriggin/dev/sean/disrust/artifacts/sustain_capture/20260317_103627/context.md)
- `qps 466734`
- `p50 1518.6us`
- `p95 3575.8us`
- `p99 6201.3us`

After:

- [context.md](/home/sriggin/dev/sean/disrust/artifacts/sustain_capture/20260317_111749/context.md)
- `qps 754068`
- `p50 1170.4us`
- `p95 1499.1us`
- `p99 2422.8us`

Interpretation:

- throughput improved by about `61.6%`
- p50, p95, and p99 all improved materially
- this is a clear win and worth keeping

Decision:

- kept
- should be committed before the next optimization pass

### Benchmark Note

The existing [pipeline_bench.rs](/home/sriggin/dev/sean/disrust/benches/pipeline_bench.rs)
remains a split submission/completion benchmark.

An attempt was made to extend it to the merged inference lane directly, but that adaptation did not
yet produce a trustworthy benchmark and was not kept. For now, optimization decisions are being
driven by:

- the sustained pipeline+writer integration tests
- the known narrow/deep and wide/shallow sustain shapes
- and `perf` on those sustain runs

### Attempt 2: Perf-Guided `Vec<PoolSlice>` Capacity Preallocation

Change:

- [submission.rs](/home/sriggin/dev/sean/disrust/src/pipeline/submission.rs)
- changed the per-batch `input_slices` construction in `build_batch_entry()` from:
  - `Vec::new()`
  to:
  - `Vec::with_capacity(max_batch_slots)`

Why this was tried:

- `perf` on the retained merged runtime showed the `inference` thread spending visible time in:
  - `malloc`
  - `realloc`
  - `alloc::raw_vec::RawVecInner<A>::finish_grow`
  - `drop_in_place<Vec<PoolSlice>>`
- so allocator churn on the per-batch `Vec<PoolSlice>` looked like a plausible low-risk target

Validation:

- [tests/submission_completion_integration.rs](/home/sriggin/dev/sean/disrust/tests/submission_completion_integration.rs)
  still passed
- `cargo clippy --all-targets --no-default-features -- -D warnings` still passed

Before / after relative to the retained scheduling-optimization baseline:

1. Narrow/deep (`2` client workers, `16` connections, `window=64`)

Before:

- [context.md](/home/sriggin/dev/sean/disrust/artifacts/sustain_capture/20260317_111652/context.md)
- `qps 983991`
- `p50 1860.6us`
- `p95 1956.9us`
- `p99 2226.2us`

After:

- [context.md](/home/sriggin/dev/sean/disrust/artifacts/sustain_capture/20260317_112207/context.md)
- `qps 987602`
- `p50 1851.4us`
- `p95 1939.5us`
- `p99 2121.7us`

Interpretation:

- slight improvement

2. Wide/shallow (`2` client workers, `250` connections, `window=2`)

Before:

- [context.md](/home/sriggin/dev/sean/disrust/artifacts/sustain_capture/20260317_111749/context.md)
- `qps 754068`
- `p50 1170.4us`
- `p95 1499.1us`
- `p99 2422.8us`

After:

- [context.md](/home/sriggin/dev/sean/disrust/artifacts/sustain_capture/20260317_112308/context.md)
- `qps 745934`
- `p50 1163.3us`
- `p95 1631.2us`
- `p99 2523.1us`

Interpretation:

- throughput regressed
- p95 and p99 regressed
- not acceptable as a retained change

Decision:

- rejected
- change backed out

### Attempt 3: Compile Out `BatchEntry` Timestamps In Non-Metrics Builds

Change:

- [batch_queue.rs](/home/sriggin/dev/sean/disrust/src/pipeline/batch_queue.rs)
- [submission.rs](/home/sriggin/dev/sean/disrust/src/pipeline/submission.rs)
- [completion.rs](/home/sriggin/dev/sean/disrust/src/pipeline/completion.rs)
- [pipeline_bench.rs](/home/sriggin/dev/sean/disrust/benches/pipeline_bench.rs)

Details:

- `BatchEntry.submitted_at` is now only present when the `metrics` feature is enabled
- the corresponding `Instant::now()` and `elapsed()` calls are compiled out of non-metrics builds

Why this was tried:

- `perf` on the retained merged runtime showed visible `clock_gettime` activity in the
  `inference` thread
- the no-CUDA sustain runs used for tuning do not enable the `metrics` feature
- so the merged lane was paying timestamp cost for bookkeeping that the build was not using

Validation:

- [tests/submission_completion_integration.rs](/home/sriggin/dev/sean/disrust/tests/submission_completion_integration.rs)
  still passes
- `cargo clippy --all-targets --no-default-features -- -D warnings` passes
- `cargo check --no-default-features --bench pipeline_bench` passes

Before / after relative to the retained scheduling-optimization baseline:

1. Narrow/deep (`2` client workers, `16` connections, `window=64`)

Before:

- [context.md](/home/sriggin/dev/sean/disrust/artifacts/sustain_capture/20260317_111652/context.md)
- `qps 983991`
- `p50 1860.6us`
- `p95 1956.9us`
- `p99 2226.2us`

After:

- [context.md](/home/sriggin/dev/sean/disrust/artifacts/sustain_capture/20260317_112609/context.md)
- `qps 1009876`
- `p50 1811.5us`
- `p95 1896.4us`
- `p99 2191.4us`

Interpretation:

- throughput improved by about `2.6%`
- p50, p95, and p99 all improved

2. Wide/shallow (`2` client workers, `250` connections, `window=2`)

Before:

- [context.md](/home/sriggin/dev/sean/disrust/artifacts/sustain_capture/20260317_111749/context.md)
- `qps 754068`
- `p50 1170.4us`
- `p95 1499.1us`
- `p99 2422.8us`

After:

- [context.md](/home/sriggin/dev/sean/disrust/artifacts/sustain_capture/20260317_112712/context.md)
- `qps 779181`
- `p50 1148.9us`
- `p95 1278.0us`
- `p99 1755.1us`

Interpretation:

- throughput improved by about `3.3%`
- p50, p95, and p99 all improved materially

Decision:

- kept
- should be committed

### Attempt 6: Compile Out `publish_to_submit` Timing In Non-Metrics Builds

Change:

- [submission.rs](/home/sriggin/dev/sean/disrust/src/pipeline/submission.rs)

Details:

- removed `PendingSlot.published_at_ns` in non-metrics builds
- compiled out the `elapsed_since_ns(next.published_at_ns)` call inside `build_batch_entry()` when
  the `metrics` feature is disabled

Why this was tried:

- `perf` on the retained baseline still showed visible `clock_gettime` activity in the merged
  inference lane
- `build_batch_entry()` still computed `publish_to_submit` timing even though
  `record_publish_to_submit()` is a no-op in non-metrics builds

Validation:

- [tests/submission_completion_integration.rs](/home/sriggin/dev/sean/disrust/tests/submission_completion_integration.rs)
  still passed
- `cargo clippy --all-targets --no-default-features -- -D warnings` passed
- `cargo check --no-default-features --bench pipeline_bench` passed

Before / after relative to the current retained baseline:

1. Narrow/deep (`2` client workers, `16` connections, `window=64`)

Before:

- [context.md](/home/sriggin/dev/sean/disrust/artifacts/sustain_capture/20260317_115439/context.md)
- `qps 1015124`
- `p50 1800.2us`
- `p95 1870.8us`
- `p99 2012.2us`

After:

- [context.md](/home/sriggin/dev/sean/disrust/artifacts/sustain_capture/20260317_115855/context.md)
- `qps 982485`
- `p50 1854.5us`
- `p95 1927.2us`
- `p99 2611.2us`

Interpretation:

- throughput regressed materially
- p50, p95, and p99 all regressed

2. Wide/shallow (`2` client workers, `250` connections, `window=2`)

Before:

- [context.md](/home/sriggin/dev/sean/disrust/artifacts/sustain_capture/20260317_115536/context.md)
- `qps 794179`
- `p50 1118.2us`
- `p95 1272.8us`
- `p99 1818.6us`

After:

- [context.md](/home/sriggin/dev/sean/disrust/artifacts/sustain_capture/20260317_115954/context.md)
- `qps 763482`
- `p50 1137.7us`
- `p95 1546.2us`
- `p99 2299.9us`

Interpretation:

- throughput regressed
- p95 and p99 regressed materially

Decision:

- rejected
- change backed out

### Attempt 7: Right-Size `ResponseFrame` Storage

Change:

- [connection_registry.rs](/home/sriggin/dev/sean/disrust/src/pipeline/connection_registry.rs)

Details:

- changed `ResponseFrame.data` from an inline `[u8; WRITE_BUF_SIZE]` to a right-sized `Box<[u8]>`
- goal was to keep stable `writev` pointers while avoiding fixed-size buffer zero-initialization for
  tiny responses

Why this was tried:

- `perf` on the retained baseline still showed inference-side `malloc`, `realloc`,
  `__memmove_avx_unaligned_erms`, and `drop_in_place<Vec<PoolSlice>>` activity
- every response still paid for a full fixed-size response buffer in the completion -> writer
  handoff, even when the actual wire frame was tiny

Validation:

- [tests/submission_completion_integration.rs](/home/sriggin/dev/sean/disrust/tests/submission_completion_integration.rs)
  still passed
- `cargo clippy --all-targets --no-default-features -- -D warnings` passed
- `cargo check --no-default-features --bench pipeline_bench` passed

Before / after relative to the current retained baseline:

1. Narrow/deep (`2` client workers, `16` connections, `window=64`)

Before:

- [context.md](/home/sriggin/dev/sean/disrust/artifacts/sustain_capture/20260317_115439/context.md)
- `qps 1015124`
- `p50 1800.2us`
- `p95 1870.8us`
- `p99 2012.2us`

After:

- [context.md](/home/sriggin/dev/sean/disrust/artifacts/sustain_capture/20260317_120208/context.md)
- `qps 1022768`
- `p50 1811.5us`
- `p95 1882.1us`
- `p99 2014.2us`

Interpretation:

- throughput improved slightly
- median and tail latency were effectively flat to slightly worse

2. Wide/shallow (`2` client workers, `250` connections, `window=2`)

Before:

- [context.md](/home/sriggin/dev/sean/disrust/artifacts/sustain_capture/20260317_115536/context.md)
- `qps 794179`
- `p50 1118.2us`
- `p95 1272.8us`
- `p99 1818.6us`

After:

- [context.md](/home/sriggin/dev/sean/disrust/artifacts/sustain_capture/20260317_120307/context.md)
- `qps 764020`
- `p50 1146.9us`
- `p95 1504.3us`
- `p99 2111.5us`

Interpretation:

- wide/shallow regressed materially
- not acceptable as a retained change

Decision:

- rejected
- change backed out

### Attempt 4: Increase Per-Pass Submission/Completion Limits To `16`

Change:

- [inference.rs](/home/sriggin/dev/sean/disrust/src/pipeline/inference.rs)
- increased:
  - `MAX_COMPLETIONS_PER_PASS` from `8` to `16`
  - `MAX_SUBMISSIONS_PER_PASS` from `8` to `16`

Why this was tried:

- after the retained `8/8` scheduling change, the next obvious scheduling question was whether the
  merged lane was still returning to the top-level loop too often under load
- if `8` was good, `16` might have reduced loop churn a little further

Validation:

- [tests/submission_completion_integration.rs](/home/sriggin/dev/sean/disrust/tests/submission_completion_integration.rs)
  still passed
- `cargo clippy --all-targets --no-default-features -- -D warnings` still passed

Before / after relative to the current retained baseline:

1. Narrow/deep (`2` client workers, `16` connections, `window=64`)

Before:

- [context.md](/home/sriggin/dev/sean/disrust/artifacts/sustain_capture/20260317_112609/context.md)
- `qps 1009876`
- `p50 1811.5us`
- `p95 1896.4us`
- `p99 2191.4us`

After:

- [context.md](/home/sriggin/dev/sean/disrust/artifacts/sustain_capture/20260317_114820/context.md)
- `qps 979785`
- `p50 1878.0us`
- `p95 1943.6us`
- `p99 1990.7us`

Interpretation:

- throughput regressed by about `3.0%`
- p50 and p95 regressed
- p99 improved slightly, but not enough to offset the throughput and median-tail loss

2. Wide/shallow (`2` client workers, `250` connections, `window=2`)

Before:

- [context.md](/home/sriggin/dev/sean/disrust/artifacts/sustain_capture/20260317_112712/context.md)
- `qps 779181`
- `p50 1148.9us`
- `p95 1278.0us`
- `p99 1755.1us`

After:

- [context.md](/home/sriggin/dev/sean/disrust/artifacts/sustain_capture/20260317_114920/context.md)
- `qps 679116`
- `p50 1183.7us`
- `p95 2041.9us`
- `p99 3958.8us`

Interpretation:

- throughput regressed badly
- p95 and p99 regressed badly
- clearly worse than the retained `8/8` setting

Decision:

- rejected
- change backed out

### Attempt 5: Compile Out In-Flight Wait Timestamps In Non-Metrics Builds

Change:

- [inference.rs](/home/sriggin/dev/sean/disrust/src/pipeline/inference.rs)

Details:

- `InflightBatchEntry.wait_started_at` is now only present when the `metrics` feature is enabled
- the `Instant::now()` taken when a batch first reports `Pending` is compiled out in non-metrics
  builds
- `metrics::record_batch_wait(...)` still receives `Duration::ZERO` in non-metrics builds so the
  callsite shape stays simple

Why this was tried:

- after compiling out `BatchEntry.submitted_at`, the merged lane still had another timestamp path
  that existed only to support metrics bookkeeping
- the no-CUDA sustain runs used for tuning do not enable `metrics`
- so the merged inference lane was still paying an avoidable `Instant::now()` cost while waiting
  on in-flight batch completion

Validation:

- [tests/submission_completion_integration.rs](/home/sriggin/dev/sean/disrust/tests/submission_completion_integration.rs)
  still passes
- `cargo clippy --all-targets --no-default-features -- -D warnings` passes

Before / after relative to the current retained baseline:

1. Narrow/deep (`2` client workers, `16` connections, `window=64`)

Before:

- [context.md](/home/sriggin/dev/sean/disrust/artifacts/sustain_capture/20260317_112609/context.md)
- `qps 1009876`
- `p50 1811.5us`
- `p95 1896.4us`
- `p99 2191.4us`

After:

- [context.md](/home/sriggin/dev/sean/disrust/artifacts/sustain_capture/20260317_115439/context.md)
- `qps 1015124`
- `p50 1800.2us`
- `p95 1870.8us`
- `p99 2012.2us`

Interpretation:

- throughput improved by about `0.5%`
- p50, p95, and p99 all improved modestly

2. Wide/shallow (`2` client workers, `250` connections, `window=2`)

Before:

- [context.md](/home/sriggin/dev/sean/disrust/artifacts/sustain_capture/20260317_112712/context.md)
- `qps 779181`
- `p50 1148.9us`
- `p95 1278.0us`
- `p99 1755.1us`

After:

- [context.md](/home/sriggin/dev/sean/disrust/artifacts/sustain_capture/20260317_115536/context.md)
- `qps 794179`
- `p50 1118.2us`
- `p95 1272.8us`
- `p99 1818.6us`

Interpretation:

- throughput improved by about `1.9%`
- p50 improved
- p95 improved slightly
- p99 regressed slightly, but the overall result is still favorable given the throughput gain and
  stable tail shape

Decision:

- kept
- should be committed

## Existing Coverage To Build On

There is already one relevant benchmark:

- [benches/pipeline_bench.rs](/home/sriggin/dev/sean/disrust/benches/pipeline_bench.rs)

What it gives us:

- a focused two-thread submission -> `BatchQueue` -> completion benchmark
- real `InferenceSession` use
- real async batch completion objects
- direct measurement of the thread-boundary cost between submission and completion

What it does **not** give us:

- pass/fail correctness coverage
- mixed progress validation
- a way to assert fairness under backlog + in-flight completions
- any writer/registry-side correctness checks

So it is useful, but it is not sufficient as the safety rail for another merge attempt.

## Recommended Test Strategy

The next merge attempt should start with a dedicated pipeline-scope test harness before changing the
runtime again.

The right scope is:

- request-ring events in
- submission and completion logic only
- no TCP
- no ingress thread
- no writer thread

That pipeline-scope harness now exists in:

- [tests/submission_completion_integration.rs](/home/sriggin/dev/sean/disrust/tests/submission_completion_integration.rs)

and it covers both:

- split `SubmissionConsumer` + `CompletionConsumer`
- merged `InferenceConsumer`

with verification of per-connection response ordering at the completion -> writer boundary.

## Sustained Writer-Backed Integration Result

The pipeline-scope test was still too small to catch the real sustain failure, so the same test
module now also contains a sustained pipeline+writer integration path using:

- real request-ring publication
- real `SubmissionConsumer` / `CompletionConsumer` or merged `InferenceConsumer`
- real `WriterConsumer`
- `UnixStream::pair()` sockets for the writer side
- clean stop signals so the threads can be joined after the test

Initial result:

- split runtime passed the sustained pipeline+writer test
- merged runtime reproduced the sustain liveness failure by stalling while publishing requests

That reproducer lived in:

- [tests/submission_completion_integration.rs](/home/sriggin/dev/sean/disrust/tests/submission_completion_integration.rs)

as:

- `split_submission_completion_and_writer_sustain_progress`
- `merged_gpu_and_writer_sustain_progress`

What this proves more precisely:

- the merged design is not merely "failing under the external sustain harness"
- it fails under an in-process sustained pipeline+writer integration workload as well
- the missing problem therefore lies beyond simple submission/completion ordering correctness

That narrowed the likely bug class to:

- progress/fairness in the merged loop under sustained load
- or interaction between the merged loop and the real writer/registry path

After the session-reservation fix described above:

- split runtime passes the sustained pipeline+writer test
- merged runtime now also passes the same sustained pipeline+writer test

So this section remains important because it documents the reproducer that exposed the bug, but the
test is now a normal passing contract test rather than a `#[should_panic]` repro.

That test should use:

- a real disruptor ring carrying `InferenceEvent`
- a real `InferenceSession` and async batch completion path
- a controllable downstream sink instead of the real writer

The sink only needs to verify:

- response order
- per-connection sequence continuity
- number of completed slots
- no lost or duplicated outputs

## Planned Integration Test

The first useful new test should be something like:

- `tests/submission_completion_integration.rs`

Suggested shape:

1. Build a small request ring with the real event type.
2. Publish synthetic `InferenceEvent`s directly into it.
3. Run the existing split submission/completion pair against that ring.
4. Replace the real completion -> writer boundary with a test sink that records:
   - connection identity
   - request sequence
   - response count
5. Assert:
   - all published requests complete
   - per-connection request sequence remains ordered
   - no progress stalls occur

This should pass against the current split-thread design first.

Only after that should a merged inference-thread implementation be introduced and held to the same test.

## Why This Test Matters

The failed wide/shallow merged run strongly suggests the missing property was not simple
"throughput" but "progress under a mixed backlog/in-flight situation."

So the test must exercise something closer to:

- backlog available
- session availability changing
- oldest batch becoming ready while new work is still arriving

That is the key behavior the previous merge attempt did not validate before full sustain runs.

## Iteration Plan

The next retry should proceed in this order:

1. Add the pipeline-scope integration test and make it pass on the current split-thread design.
2. Keep [benches/pipeline_bench.rs](/home/sriggin/dev/sean/disrust/benches/pipeline_bench.rs) as
   the low-level cost benchmark for the existing boundary.
3. Add the missing merged-thread instrumentation:
   - in-flight batch count
   - oldest in-flight age
   - ready-but-not-retired delay
4. Implement a merged inference-thread scheduler with an explicit fairness policy.
5. Run the new integration test.
6. Run the existing sustain shapes:
   - narrow/deep: `2` client threads, `16` connections, `window=64`
   - wide/shallow: `2` client threads, `250` connections, `window=2`
7. Keep the merge only if:
   - the new integration test passes
   - both sustain shapes complete correctly
   - performance is at least neutral on the previously healthy shape

## Current State

The repository currently contains:

- the merged `InferenceConsumer` implementation
- the existing split `SubmissionConsumer` / `CompletionConsumer`
- the new isolated submission/completion integration test

At the time of this note, the server runtime may be wired either way during active experimentation,
but the measurements above show the merged runtime is not currently acceptable for the real
benchmark shapes already in use.
