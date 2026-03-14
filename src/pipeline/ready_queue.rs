//! SPSC ready queue for connection notifications from completion to writer.

use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConnectionRef {
    pub conn_id: u16,
    pub generation: u32,
}

pub struct ReadyQueue {
    capacity: usize,
    head: AtomicUsize,
    tail: AtomicUsize,
    slots: Box<[UnsafeCell<MaybeUninit<ConnectionRef>>]>,
}

unsafe impl Send for ReadyQueue {}
unsafe impl Sync for ReadyQueue {}

impl ReadyQueue {
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

    pub fn push(&self, entry: ConnectionRef) {
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

    pub fn pop(&self) -> Option<ConnectionRef> {
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

impl Drop for ReadyQueue {
    fn drop(&mut self) {
        while self.pop().is_some() {}
    }
}

#[cfg(test)]
mod tests {
    use super::{ConnectionRef, ReadyQueue};

    #[test]
    fn preserves_fifo_order() {
        let queue = ReadyQueue::new(4);
        queue.push(ConnectionRef {
            conn_id: 1,
            generation: 11,
        });
        queue.push(ConnectionRef {
            conn_id: 2,
            generation: 22,
        });
        assert_eq!(
            queue.pop(),
            Some(ConnectionRef {
                conn_id: 1,
                generation: 11,
            })
        );
        assert_eq!(
            queue.pop(),
            Some(ConnectionRef {
                conn_id: 2,
                generation: 22,
            })
        );
        assert_eq!(queue.pop(), None);
    }
}
