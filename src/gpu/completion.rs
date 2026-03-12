//! CompletionConsumer: waits for async GPU batch completion, encodes responses, writes directly
//! to client fds via its own io_uring ring.

use std::io;
use std::sync::Arc;

use disruptor::{EventGuard, EventPoller, Polling, SingleConsumerBarrier};
use io_uring::{opcode, squeue::Entry, types::Fd};

use crate::batch_queue::BatchQueue;
use crate::config::{MAX_SESSION_BATCH_SIZE, SESSION_POOL_SIZE, WRITE_BUF_SIZE, WRITE_BUF_SLOTS};
use crate::gpu::diag::{
    self, BATCHES_COMPLETED, RESPONSES_WRITTEN, WRITE_CQES, WRITE_NEGATIVE, WRITE_SQES,
};
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
    diag::bump(&WRITE_CQES, 1);

    let meta = write_meta[buf_idx];
    if meta.expected_len == 0 {
        return;
    }

    if result == meta.expected_len as i32 {
        return;
    }

    if result < 0 {
        diag::bump(&WRITE_NEGATIVE, 1);
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
    max_batch_slots: usize,
}

// SAFETY: CompletionConsumer runs on a single dedicated thread.
unsafe impl Send for CompletionConsumer {}

impl CompletionConsumer {
    pub fn new(
        poller: EventPoller<InferenceEvent, SingleConsumerBarrier>,
        batch_queue: Arc<BatchQueue>,
        max_batch_slots: usize,
    ) -> io::Result<Self> {
        assert!(max_batch_slots > 0, "max_batch_slots must be > 0");
        assert!(
            max_batch_slots <= MAX_SESSION_BATCH_SIZE,
            "max_batch_slots ({max_batch_slots}) exceeds compile-time limit ({MAX_SESSION_BATCH_SIZE})"
        );
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
            max_batch_slots,
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

            let entry = loop {
                if let Some(entry) = self.batch_queue.pop() {
                    break entry;
                }
                std::hint::spin_loop();
            };

            entry.batch.completion.wait();

            let mut guard = loop {
                match self.poller.poll_take(entry.slot_count as u64) {
                    Ok(guard) => break guard,
                    Err(Polling::NoEvents) => std::hint::spin_loop(),
                    Err(Polling::Shutdown) => return,
                }
            };

            process_batch(
                &mut guard,
                entry,
                &mut self.ring,
                &mut self.write_bufs,
                &mut self.outstanding,
                &mut self.write_meta,
                &mut self.active_set,
                self.max_batch_slots,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn process_batch(
    guard: &mut EventGuard<'_, InferenceEvent, SingleConsumerBarrier>,
    entry: crate::batch_queue::BatchEntry,
    ring: &mut IoUring,
    write_bufs: &mut Box<[[u8; WRITE_BUF_SIZE]; WRITE_BUF_SLOTS]>,
    outstanding: &mut [usize; 2],
    write_meta: &mut [WriteMeta; WRITE_BUF_SLOTS],
    active_set: &mut usize,
    max_batch_slots: usize,
) {
    let mut guard_ref = &mut *guard;
    let mut slot_in_set: usize = 0;

    let rotate_set = |ring: &mut IoUring,
                      outstanding: &mut [usize; 2],
                      write_meta: &[WriteMeta; WRITE_BUF_SLOTS],
                      active_set: &mut usize,
                      slot_in_set: &mut usize| {
        if *slot_in_set == 0 {
            return;
        }
        ring.submit();
        ring.drain_until_zero(outstanding, write_meta, *active_set);
        *active_set ^= 1;
        *slot_in_set = 0;
    };

    let output =
        unsafe { std::slice::from_raw_parts(entry.batch.output_ptr, entry.batch.output_len) };
    let mut output_offset: usize = 0;

    for _ in 0..entry.slot_count {
        if slot_in_set >= max_batch_slots {
            rotate_set(ring, outstanding, write_meta, active_set, &mut slot_in_set);
        }

        let event = guard_ref
            .next()
            .expect("guard exhausted before queued batch slot_count");
        let num_vecs = event.num_vectors as usize;
        let fd = event.fd;

        // Encode wire format: [u8 num_vectors][f32 × num_vectors LE].
        let buf_idx = (*active_set * MAX_SESSION_BATCH_SIZE) + slot_in_set;
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
        outstanding[*active_set] += 1;
        diag::bump(&WRITE_SQES, 1);

        output_offset += num_vecs;
        slot_in_set += 1;
        diag::bump(&RESPONSES_WRITTEN, 1);
        crate::metrics::dec_req_occ();
    }

    rotate_set(ring, outstanding, write_meta, active_set, &mut slot_in_set);

    debug_assert_eq!(
        output_offset, entry.batch.output_len,
        "slot_count/num_vectors mismatch between ring events and batch output"
    );

    let session_available = Arc::clone(&entry.batch.session_available);
    drop(entry);
    session_available.store(true, std::sync::atomic::Ordering::Release);
    diag::bump(&BATCHES_COMPLETED, 1);
}
