use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::os::fd::RawFd;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::config::WRITE_BUF_SIZE;
use crate::connection_id::ConnectionRef;
use crate::protocol;

#[derive(Clone, Copy)]
pub struct ResponseReady {
    pub conn: ConnectionRef,
    pub request_seq: u64,
    pub published_at_ns: u64,
    pub len: usize,
    pub data: [u8; WRITE_BUF_SIZE],
}

impl ResponseReady {
    pub fn encode(
        conn: ConnectionRef,
        request_seq: u64,
        published_at_ns: u64,
        results: &[f32],
    ) -> Self {
        let len = protocol::response_size(results.len());
        debug_assert!(len <= WRITE_BUF_SIZE);
        let mut data = [0u8; WRITE_BUF_SIZE];
        protocol::encode_response(results, &mut data[..len]);
        Self {
            conn,
            request_seq,
            published_at_ns,
            len,
            data,
        }
    }
}

pub struct ResponseQueue {
    capacity: usize,
    head: AtomicUsize,
    tail: AtomicUsize,
    notify_fd: RawFd,
    slots: Box<[UnsafeCell<MaybeUninit<ResponseReady>>]>,
}

unsafe impl Send for ResponseQueue {}
unsafe impl Sync for ResponseQueue {}

impl ResponseQueue {
    pub fn new(capacity: usize) -> Self {
        let notify_fd = unsafe { libc::eventfd(0, libc::EFD_NONBLOCK | libc::EFD_CLOEXEC) };
        assert!(notify_fd >= 0, "eventfd creation failed");
        let slots = (0..capacity)
            .map(|_| UnsafeCell::new(MaybeUninit::uninit()))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self {
            capacity,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
            notify_fd,
            slots,
        }
    }

    pub fn push(&self, entry: ResponseReady) {
        loop {
            let tail = self.tail.load(Ordering::Relaxed);
            let head = self.head.load(Ordering::Acquire);
            if tail.wrapping_sub(head) < self.capacity {
                let was_empty = tail == head;
                let idx = tail % self.capacity;
                unsafe { (*self.slots[idx].get()).write(entry) };
                self.tail.store(tail.wrapping_add(1), Ordering::Release);
                if was_empty {
                    let one = 1u64;
                    let rc = unsafe {
                        libc::write(
                            self.notify_fd,
                            (&one as *const u64).cast::<libc::c_void>(),
                            std::mem::size_of::<u64>(),
                        )
                    };
                    if rc >= 0 {
                        assert_eq!(
                            rc as usize,
                            std::mem::size_of::<u64>(),
                            "short eventfd write"
                        );
                    }
                }
                return;
            }
            std::hint::spin_loop();
        }
    }

    pub fn notify_fd(&self) -> RawFd {
        self.notify_fd
    }

    pub fn pop(&self) -> Option<ResponseReady> {
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

impl Drop for ResponseQueue {
    fn drop(&mut self) {
        while self.pop().is_some() {}
        if self.notify_fd >= 0 {
            unsafe {
                libc::close(self.notify_fd);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ResponseQueue, ResponseReady};
    use crate::connection_id::ConnectionRef;

    #[test]
    fn preserves_fifo_order() {
        let queue = ResponseQueue::new(4);
        let conn0 = ConnectionRef::new(0, 1, 11);
        let conn1 = ConnectionRef::new(1, 2, 22);
        queue.push(ResponseReady::encode(conn0, 7, 100, &[1.0f32, 2.0, 3.0]));
        queue.push(ResponseReady::encode(conn1, 8, 101, &[4.0f32, 5.0]));

        let first = queue.pop().expect("first response");
        assert_eq!(first.conn, conn0);
        assert_eq!(first.request_seq, 7);
        assert_eq!(first.data[0], 3); // num_vectors header

        let second = queue.pop().expect("second response");
        assert_eq!(second.conn, conn1);
        assert_eq!(second.request_seq, 8);
        assert_eq!(second.data[0], 2); // num_vectors header

        assert!(queue.pop().is_none());
    }
}
