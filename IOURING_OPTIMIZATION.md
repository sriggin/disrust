# io_uring Optimization Opportunities

Current state: the ingress IO thread uses three ops (`OP_ACCEPT`, `Recv`, `Writev`) and one
cross-thread notification (`PollAdd` on an eventfd). Ring setup already enables
`IORING_SETUP_SINGLE_ISSUER`, `IORING_SETUP_COOP_TASKRUN`, and
`IORING_SETUP_DEFER_TASKRUN`. The following optimizations remain available, ordered by effort.

---

## Low effort: multishot accept

**Current behaviour.** `handle_accept` resubmits a fresh `OP_ACCEPT` SQE after every
accepted-connection CQE. One connection = one SQE consumed.

**Change.** Set `IORING_ACCEPT_MULTISHOT` on the accept SQE (kernel 5.19+). The SQE stays
live and emits one CQE per accepted connection indefinitely. The CQE handling needs a guard
to detect the `IORING_CQE_F_MORE` flag and skip re-submission while the SQE is still active.
This is now mostly local to the accept submission/handler path because the ingress wrapper
already surfaces CQE flags, but it is not just a one-line opcode tweak.

**Removes.** The entire accept-resubmit path in `handle_accept` and the associated SQE
budget consumption.

---

## Medium effort: registered files

**Current behaviour.** Every `Recv` and `Writev` SQE carries a raw fd. The kernel looks it
up in the process file descriptor table on each operation.

**Change.** Register the connection fd array with `IORING_REGISTER_FILES` at startup and on
each accept/close, then set `IOSQE_FIXED_FILE` on all read/write SQEs using the slot index
instead of the raw fd.

**Requires.** A parallel "registered fd slot" alongside each `Slab` slot, updated on
`handle_accept` (register) and connection reap (unregister via `IORING_UNREGISTER_FILES` or
slot update to `-1`).

**Removes.** File table lookup on every IO op. Meaningful at high connection counts where
many fds are active concurrently.

---

## Medium effort: registered buffers for response writes

**Current behaviour.** Each `Writev` SQE points into `ResponseFrame::data` inline arrays
inside `Box<ResponseFrame>` allocations. The kernel re-pins these memory pages on every
write syscall.

**Change.** Pre-allocate a pool of response buffers with stable addresses and register that
memory with `IORING_REGISTER_BUFFERS`. Then redesign the write path to submit from registered
memory directly, likely via `WriteFixed` on contiguous registered buffers or via a coalesced
response representation. This is not a flag-only change to the current `Writev` path.

**Requires.** Replacing `VecDeque<Box<ResponseFrame>>` per connection with indices into a
shared pre-allocated frame pool, plus changing the current vectored write batching strategy.
Frame lifecycle (alloc on response arrival, free on write completion) mirrors the existing
`BufferPool` pattern, but the submission opcode/layout also has to change.

**Removes.** Kernel re-pinning overhead on every write. Also reduces allocator pressure by
replacing per-response `Box` allocations with pool indices.

---

## High effort: multishot recv + buffer rings

This is the highest-value structural change. It eliminates the dominant per-connection memory
cost and the read resubmit path.

### Problem: 65KB per connection

Every `Connection` owns `Box<[u8; READ_BUF_SIZE]>` (65 536 bytes). At `SLAB_CAPACITY = 4096`
connections per IO thread, that is **256 MB of read buffer memory per thread**, all of it
resident and hot even for idle connections.

### Buffer rings (`IORING_REGISTER_PBUF_RING`, kernel 5.19+)

Register a shared pool of fixed-size buffers with the kernel. On each `Recv` SQE, omit the
buffer address and set `IOSQE_BUFFER_SELECT` with a buffer group ID. The kernel picks an
available buffer at receive time and returns its ID in the CQE flags. The application
processes the buffer then returns it to the ring via `io_uring_buf_ring_add`.

This makes the 65KB per-connection allocation disappear. Buffer memory is shared across all
connections and sized to the expected concurrency, not the maximum connection count.

### Multishot recv (`IORING_RECV_MULTISHOT`, kernel 5.19+, requires buffer rings)

Set `IORING_RECV_MULTISHOT` on the `Recv` SQE. Like multishot accept, the SQE stays live
and emits a CQE each time data arrives on the socket. This removes the per-read resubmit
path, but it does not remove the need for per-connection parse carry state.

### Impact on connection state machine

- The fixed 64 KiB per-connection `read_buf` allocation can be removed.
- The current `read_inflight` boolean can be removed or reduced to multishot-liveness state
  tracked at the ring/handler level rather than a per-read resubmit guard.
- Parse input comes from the buffer ring slot identified by the CQE, but incomplete requests
  still need per-connection carry state.
- Partial-read compaction does not disappear; it moves into explicit carry handling: either
  copy unconsumed bytes into per-connection scratch space before returning the buffer, or
  retain ownership of a provided buffer across CQEs until the parser has consumed it.
- Buffer ring size must be tuned to expected concurrent in-flight reads; exhaustion stalls
  reads until a buffer is returned.

---

## Low priority: SQ polling (`IORING_SETUP_SQPOLL`)

A kernel thread polls the SQ ring, eliminating the `io_uring_enter` syscall for submissions.
The existing parse-budget batching already reduces syscall frequency significantly. SQPOLL
consumes a dedicated kernel CPU and requires tuning of the idle timeout. Profile submission
syscall cost before considering this; it is unlikely to be the bottleneck while GPU inference
latency dominates.

---

## Implementation order

1. Multishot accept — now mostly isolated to the accept submission/handler path because CQE
   flags are already surfaced through the ingress wrapper.
2. Registered files — requires slab/accept/reap coordination but no memory layout changes.
3. Registered buffers for writes — requires response frame pool plus a write-path redesign;
   can still be done independently of the read path.
4. Multishot recv + buffer rings — the largest change; touches connection state, parse loop,
   carry handling, and buffer lifecycle.
5. SQPOLL — only after profiling confirms submission syscalls are measurable.
