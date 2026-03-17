//! SPSC batch queue between SubmissionConsumer and CompletionConsumer.
//!
//! `BatchEntry` contains completion state and backend resource handles that
//! cannot sit in a pre-allocated ring slot, which is why this queue exists at all.

use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(feature = "metrics")]
use std::time::Instant;

use crate::pipeline::session::InFlightBatch;

/// One entry in the batch queue.
pub struct BatchEntry<R: Send> {
    /// Number of disruptor slots covered by this batch.
    pub slot_count: usize,
    /// Wall-clock timestamp when submission handed this batch to completion.
    #[cfg(feature = "metrics")]
    pub submitted_at: Instant,
    /// In-flight batch state. Submission enqueues this immediately after
    /// the backend submit call returns; completion waits for the signal.
    pub batch: InFlightBatch<R>,
}

/// Fixed-capacity SPSC ring for `BatchEntry` values.
///
/// - **Producer** (SubmissionConsumer): calls `push`, spins on full.
/// - **Consumer** (CompletionConsumer): calls `pop`, returns `None` when empty.
///
/// The `+1` over `SESSION_POOL_SIZE` in `BATCH_QUEUE_CAPACITY` prevents a wrap-induced
/// deadlock where SubmissionConsumer needs to enqueue a second batch from one poll cycle
/// before CompletionConsumer can advance the sequence barrier.
pub struct BatchQueue<R: Send> {
    capacity: usize,
    /// Read cursor - advanced by the consumer.
    head: AtomicUsize,
    /// Write cursor - advanced by the producer.
    tail: AtomicUsize,
    slots: Box<[UnsafeCell<MaybeUninit<BatchEntry<R>>>]>,
}

unsafe impl<R: Send> Send for BatchQueue<R> {}
unsafe impl<R: Send> Sync for BatchQueue<R> {}

impl<R: Send> BatchQueue<R> {
    pub fn new(capacity: usize) -> Self {
        let slots = (0..capacity)
            .map(|_| UnsafeCell::new(MaybeUninit::uninit()))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self {
            capacity,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
            slots,
        }
    }

    /// Push an entry, spinning until space is available.
    pub fn push(&self, entry: BatchEntry<R>) {
        loop {
            let tail = self.tail.load(Ordering::Relaxed);
            let head = self.head.load(Ordering::Acquire);
            if tail.wrapping_sub(head) < self.capacity {
                let idx = tail % self.capacity;
                unsafe { (*self.slots[idx].get()).write(entry) };
                self.tail.store(tail.wrapping_add(1), Ordering::Release);
                return;
            }
            std::hint::spin_loop();
        }
    }

    /// Pop an entry. Returns `None` when the queue is empty.
    pub fn pop(&self) -> Option<BatchEntry<R>> {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);
        if head == tail {
            return None;
        }
        let idx = head % self.capacity;
        let entry = unsafe { (*self.slots[idx].get()).assume_init_read() };
        self.head.store(head.wrapping_add(1), Ordering::Release);
        Some(entry)
    }
}

impl<R: Send> Drop for BatchQueue<R> {
    fn drop(&mut self) {
        while self.pop().is_some() {}
    }
}
