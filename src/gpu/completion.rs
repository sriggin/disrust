//! CompletionConsumer: waits for async GPU batch completion, encodes responses, writes directly
//! to client fds via its own io_uring ring.

use std::io;
use std::sync::Arc;

use disruptor::{EventGuard, EventPoller, Polling, SingleConsumerBarrier};
use io_uring::{opcode, squeue::Entry, types::Fd};

use crate::batch_queue::BatchQueue;
use crate::config::{MAX_SESSION_BATCH_SIZE, SESSION_POOL_SIZE, WRITE_BUF_SIZE, WRITE_BUF_SLOTS};
use crate::ring_types::InferenceEvent;

struct IoUring {
    inner: io_uring::IoUring,
}

impl IoUring {
    fn new(entries: u32) -> io::Result<Self> {
        Ok(Self {
            inner: io_uring::IoUring::new(entries)?,
        })
    }

    fn push(&mut self, sqe: &Entry) {
        loop {
            match unsafe { self.inner.submission().push(sqe) } {
                Ok(()) => return,
                Err(_) => {
                    self.inner.submit().expect("SQ flush failed");
                }
            }
        }
    }

    fn submit(&mut self) {
        self.inner.submit().expect("io_uring submit failed");
    }

    fn drain_cqes(&mut self, mut on_cqe: impl FnMut(u64, i32)) {
        for cqe in self.inner.completion() {
            on_cqe(cqe.user_data(), cqe.result());
        }
    }

    /// Drain CQEs until `outstanding[set]` reaches 0.
    fn drain_until_zero(
        &mut self,
        outstanding: &mut [usize; 2],
        write_meta: &[WriteMeta; WRITE_BUF_SLOTS],
        set: usize,
    ) {
        while outstanding[set] > 0 {
            self.inner
                .submit_and_wait(1)
                .expect("submit_and_wait failed");
            for cqe in self.inner.completion() {
                handle_write_cqe(cqe.user_data(), cqe.result(), outstanding, write_meta);
            }
        }
    }
}

#[derive(Clone, Copy, Default)]
struct WriteMeta {
    fd: i32,
    expected_len: u32,
}

fn handle_write_cqe(
    user_data: u64,
    result: i32,
    outstanding: &mut [usize; 2],
    write_meta: &[WriteMeta; WRITE_BUF_SLOTS],
) {
    let buf_idx = user_data as usize;
    let buf_set = buf_idx / MAX_SESSION_BATCH_SIZE;
    outstanding[buf_set] = outstanding[buf_set].saturating_sub(1);

    let meta = write_meta[buf_idx];
    if meta.expected_len == 0 {
        return;
    }

    if result == meta.expected_len as i32 {
        return;
    }

    if result < 0 {
        match -result {
            libc::EPIPE | libc::EBADF | libc::ECONNRESET => {}
            libc::EAGAIN => unsafe {
                libc::shutdown(meta.fd, libc::SHUT_WR);
            },
            _ => {}
        }
        return;
    }

    // Partial writes leave a corrupt frame on the stream. Shut down the write side so the
    // reader observes EOF and normal cleanup runs on the IO thread.
    unsafe {
        libc::shutdown(meta.fd, libc::SHUT_WR);
    }
}

pub struct CompletionConsumer {
    poller: EventPoller<InferenceEvent, SingleConsumerBarrier>,
    batch_queue: Arc<BatchQueue>,
    ring: IoUring,
    /// Double-buffered write buffers: [WRITE_BUF_SLOTS][WRITE_BUF_SIZE].
    write_bufs: Box<[[u8; WRITE_BUF_SIZE]; WRITE_BUF_SLOTS]>,
    /// Outstanding OP_WRITE counts per buffer set (A=0, B=1).
    outstanding: [usize; 2],
    /// Per-buffer metadata for CQE result handling.
    write_meta: [WriteMeta; WRITE_BUF_SLOTS],
    /// Currently active write buffer set.
    active_set: usize,
    /// Absolute ring sequence of the last slot fully processed.
    local_cursor: i64,
}

// SAFETY: CompletionConsumer runs on a single dedicated thread.
unsafe impl Send for CompletionConsumer {}

impl CompletionConsumer {
    pub fn new(
        poller: EventPoller<InferenceEvent, SingleConsumerBarrier>,
        batch_queue: Arc<BatchQueue>,
    ) -> io::Result<Self> {
        let sq_depth = (SESSION_POOL_SIZE * MAX_SESSION_BATCH_SIZE) as u32;
        let ring = IoUring::new(sq_depth)?;
        Ok(Self {
            poller,
            batch_queue,
            ring,
            write_bufs: Box::new([[0u8; WRITE_BUF_SIZE]; WRITE_BUF_SLOTS]),
            outstanding: [0, 0],
            write_meta: [WriteMeta::default(); WRITE_BUF_SLOTS],
            active_set: 0,
            local_cursor: -1,
        })
    }

    pub fn run(mut self) {
        loop {
            // Phase A: ensure the previous buffer set is fully retired.
            let prev_set = 1 - self.active_set;
            self.ring
                .drain_until_zero(&mut self.outstanding, &self.write_meta, prev_set);

            // Drain any stray CQEs for the current set.
            {
                let outstanding = &mut self.outstanding;
                let write_meta = &self.write_meta;
                self.ring.drain_cqes(|user_data, result| {
                    handle_write_cqe(user_data, result, outstanding, write_meta);
                });
            }

            // Phase B: poll for new events.
            match self.poller.poll() {
                Ok(mut guard) => {
                    process_guard(
                        &mut guard,
                        &self.batch_queue,
                        &mut self.ring,
                        &mut self.write_bufs,
                        &mut self.outstanding,
                        &mut self.write_meta,
                        self.active_set,
                        &mut self.local_cursor,
                    );
                    self.ring.submit();
                    // Guard drop advances CompletionConsumer cursor.
                    self.active_set ^= 1;
                }
                Err(Polling::NoEvents) => std::hint::spin_loop(),
                Err(Polling::Shutdown) => return,
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn process_guard(
    guard: &mut EventGuard<'_, InferenceEvent, SingleConsumerBarrier>,
    batch_queue: &Arc<BatchQueue>,
    ring: &mut IoUring,
    write_bufs: &mut Box<[[u8; WRITE_BUF_SIZE]; WRITE_BUF_SLOTS]>,
    outstanding: &mut [usize; 2],
    write_meta: &mut [WriteMeta; WRITE_BUF_SLOTS],
    active_set: usize,
    local_cursor: &mut i64,
) {
    let mut guard_ref = &mut *guard;
    let buf_base = active_set * MAX_SESSION_BATCH_SIZE;
    let mut slot_in_set: usize = 0;

    loop {
        // Pop next BatchEntry — spin until the SubmissionConsumer has pushed it.
        let entry = loop {
            if let Some(e) = batch_queue.pop() {
                break e;
            }
            std::hint::spin_loop();
        };

        debug_assert!(
            entry.end_sequence as i64 > *local_cursor,
            "BatchEntry.end_sequence ({}) must be > local_cursor ({})",
            entry.end_sequence,
            *local_cursor
        );

        entry.batch.completion.wait();
        let output =
            unsafe { std::slice::from_raw_parts(entry.batch.output_ptr, entry.batch.output_len) };
        let mut output_offset: usize = 0;

        let slots_in_batch = entry.end_sequence as i64 - *local_cursor;
        for _ in 0..slots_in_batch {
            let event = guard_ref
                .next()
                .expect("guard exhausted before BatchEntry.end_sequence");
            let num_vecs = event.num_vectors as usize;
            let fd = event.fd;

            // Encode wire format: [u8 num_vectors][f32 × num_vectors LE].
            let buf_idx = buf_base + slot_in_set;
            let buf = &mut write_bufs[buf_idx];
            buf[0] = num_vecs as u8;
            for (i, &val) in output[output_offset..output_offset + num_vecs]
                .iter()
                .enumerate()
            {
                buf[1 + i * 4..1 + i * 4 + 4].copy_from_slice(&val.to_le_bytes());
            }
            let wire_len = (1 + num_vecs * 4) as u32;

            // Submit OP_WRITE (user_data encodes the active set for CQE accounting).
            write_meta[buf_idx] = WriteMeta {
                fd,
                expected_len: wire_len,
            };
            let sqe = opcode::Write::new(Fd(fd), buf.as_ptr(), wire_len)
                .build()
                .user_data(buf_idx as u64);
            ring.push(&sqe);
            outstanding[active_set] += 1;

            // Eagerly release the PoolSlice — GPU has finished reading this slot's input.
            event.features.release();

            output_offset += num_vecs;
            slot_in_set += 1;
        }

        *local_cursor = entry.end_sequence as i64;
        let session_available = Arc::clone(&entry.batch.session_available);
        drop(entry);
        session_available.store(true, std::sync::atomic::Ordering::Release);

        // Exit when the guard is fully consumed.
        if guard_ref.len() == 0 {
            break;
        }
    }
}
