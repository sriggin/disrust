use std::collections::VecDeque;
use std::os::fd::RawFd;
use std::sync::Mutex;

use crate::clock::elapsed_since_ns;
use crate::config::{SLAB_CAPACITY, WRITE_BUF_SIZE};
use crate::metrics;
use crate::pipeline::ready_queue::ConnectionRef;

const MAX_IOVECS_PER_WRITE: usize = 64;

pub struct ResponseFrame {
    pub request_seq: u64,
    pub published_at_ns: u64,
    pub len: usize,
    pub offset: usize,
    pub data: [u8; WRITE_BUF_SIZE],
}

impl ResponseFrame {
    pub fn new(request_seq: u64, published_at_ns: u64, bytes: &[u8]) -> Self {
        debug_assert!(bytes.len() <= WRITE_BUF_SIZE);
        let mut data = [0u8; WRITE_BUF_SIZE];
        data[..bytes.len()].copy_from_slice(bytes);
        Self {
            request_seq,
            published_at_ns,
            len: bytes.len(),
            offset: 0,
            data,
        }
    }

    fn remaining(&self) -> usize {
        self.len.saturating_sub(self.offset)
    }

    fn remaining_ptr(&self) -> *const u8 {
        unsafe { self.data.as_ptr().add(self.offset) }
    }
}

struct ConnectionWriteState {
    generation: u32,
    fd: RawFd,
    read_closed: bool,
    write_closed: bool,
    retired: bool,
    published_seq_end: u64,
    completed_seq_end: u64,
    ready_queued: bool,
    write_inflight: bool,
    queue: VecDeque<Box<ResponseFrame>>,
    inflight: VecDeque<Box<ResponseFrame>>,
    inflight_iovecs: [libc::iovec; MAX_IOVECS_PER_WRITE],
    inflight_iov_count: usize,
}

impl ConnectionWriteState {
    fn new() -> Self {
        Self {
            generation: 0,
            fd: -1,
            read_closed: false,
            write_closed: false,
            retired: true,
            published_seq_end: 0,
            completed_seq_end: 0,
            ready_queued: false,
            write_inflight: false,
            queue: VecDeque::new(),
            inflight: VecDeque::new(),
            inflight_iovecs: [libc::iovec {
                iov_base: std::ptr::null_mut(),
                iov_len: 0,
            }; MAX_IOVECS_PER_WRITE],
            inflight_iov_count: 0,
        }
    }

    fn open(&mut self, fd: RawFd) -> u32 {
        self.generation = self.generation.wrapping_add(1).max(1);
        self.fd = fd;
        self.read_closed = false;
        self.write_closed = false;
        self.retired = false;
        self.published_seq_end = 0;
        self.completed_seq_end = 0;
        self.ready_queued = false;
        self.write_inflight = false;
        self.queue.clear();
        self.inflight.clear();
        self.inflight_iov_count = 0;
        self.generation
    }

    fn matches(&self, generation: u32) -> bool {
        !self.retired && self.generation == generation
    }

    fn maybe_retire(&mut self) {
        if self.retired {
            return;
        }
        if self.read_closed
            && !self.write_inflight
            && self.queue.is_empty()
            && self.inflight.is_empty()
            && self.completed_seq_end >= self.published_seq_end
        {
            if self.fd >= 0 {
                unsafe {
                    libc::close(self.fd);
                }
            }
            self.fd = -1;
            self.retired = true;
            self.write_closed = true;
            self.ready_queued = false;
            self.inflight_iov_count = 0;
        }
    }
}

struct Slot {
    state: Mutex<ConnectionWriteState>,
}

pub struct ConnectionRegistry {
    slots: Box<[Slot]>,
}

impl ConnectionRegistry {
    pub fn new(capacity: usize) -> Self {
        let slots = (0..capacity)
            .map(|_| Slot {
                state: Mutex::new(ConnectionWriteState::new()),
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self { slots }
    }

    pub fn open(&self, conn_id: u16, fd: RawFd) -> u32 {
        let mut state = self.slots[conn_id as usize].state.lock().unwrap();
        assert!(
            state.retired,
            "connection slot {} reopened before retirement",
            conn_id
        );
        state.open(fd)
    }

    pub fn update_published_seq_end(&self, conn_id: u16, generation: u32, published_seq_end: u64) {
        let mut state = self.slots[conn_id as usize].state.lock().unwrap();
        if state.matches(generation) {
            state.published_seq_end = published_seq_end.max(state.published_seq_end);
        }
    }

    pub fn mark_read_closed(&self, conn_id: u16, generation: u32, published_seq_end: u64) {
        let mut state = self.slots[conn_id as usize].state.lock().unwrap();
        if !state.matches(generation) {
            return;
        }
        state.read_closed = true;
        state.published_seq_end = published_seq_end.max(state.published_seq_end);
        state.maybe_retire();
    }

    pub fn is_retired(&self, conn_id: u16, generation: u32) -> bool {
        let state = self.slots[conn_id as usize].state.lock().unwrap();
        state.generation == generation && state.retired
    }

    pub fn enqueue_response(
        &self,
        conn: ConnectionRef,
        request_seq: u64,
        published_at_ns: u64,
        bytes: &[u8],
    ) -> bool {
        let mut state = self.slots[conn.conn_id as usize].state.lock().unwrap();
        if !state.matches(conn.generation) || state.write_closed {
            return false;
        }
        state.queue.push_back(Box::new(ResponseFrame::new(
            request_seq,
            published_at_ns,
            bytes,
        )));
        if state.ready_queued {
            false
        } else {
            state.ready_queued = true;
            true
        }
    }

    pub fn take_flush(&self, conn: ConnectionRef) -> Option<FlushSubmission> {
        let mut state = self.slots[conn.conn_id as usize].state.lock().unwrap();
        if !state.matches(conn.generation) {
            return None;
        }

        state.ready_queued = false;
        if state.write_closed || state.write_inflight {
            return None;
        }

        if state.inflight.is_empty() {
            while state.inflight.len() < MAX_IOVECS_PER_WRITE {
                let Some(frame) = state.queue.pop_front() else {
                    break;
                };
                metrics::record_publish_to_write_submit(elapsed_since_ns(frame.published_at_ns));
                state.inflight.push_back(frame);
            }
        }

        if state.inflight.is_empty() {
            state.maybe_retire();
            return None;
        }

        let mut iov_meta = [(std::ptr::null_mut(), 0usize); MAX_IOVECS_PER_WRITE];
        let mut iov_count = 0usize;
        for frame in state.inflight.iter() {
            iov_meta[iov_count] = (
                frame.remaining_ptr() as *mut libc::c_void,
                frame.remaining(),
            );
            iov_count += 1;
        }
        state.inflight_iov_count = iov_count;
        for (idx, (base, len)) in iov_meta.into_iter().take(iov_count).enumerate() {
            state.inflight_iovecs[idx] = libc::iovec {
                iov_base: base,
                iov_len: len,
            };
        }
        state.write_inflight = true;

        Some(FlushSubmission {
            fd: state.fd,
            iovecs: state.inflight_iovecs.as_ptr(),
            iov_count: state.inflight_iov_count as u32,
        })
    }

    pub fn handle_write_result(&self, conn: ConnectionRef, result: i32) -> WriteResult {
        let mut state = self.slots[conn.conn_id as usize].state.lock().unwrap();
        if !state.matches(conn.generation) {
            return WriteResult::Stale;
        }
        if !state.write_inflight {
            return WriteResult::Idle;
        }

        if result < 0 {
            state.write_closed = true;
            state.write_inflight = false;
            state.inflight.clear();
            state.queue.clear();
            state.inflight_iov_count = 0;
            state.read_closed = true;
            state.published_seq_end = state.completed_seq_end;
            state.maybe_retire();
            return WriteResult::Error(-result);
        }

        let mut remaining = result as usize;
        while remaining > 0 {
            let Some(frame) = state.inflight.front_mut() else {
                break;
            };
            let frame_remaining = frame.remaining();
            if remaining >= frame_remaining {
                remaining -= frame_remaining;
                state.completed_seq_end = frame.request_seq + 1;
                state.inflight.pop_front();
            } else {
                frame.offset += remaining;
                remaining = 0;
            }
        }

        state.write_inflight = false;
        state.inflight_iov_count = 0;

        if !state.inflight.is_empty() {
            state.write_inflight = false;
            return WriteResult::NeedsResubmit;
        }

        if !state.queue.is_empty() {
            state.ready_queued = true;
            return WriteResult::ReadyAgain;
        }

        state.maybe_retire();
        WriteResult::Completed
    }

    pub fn current_fd(&self, conn: ConnectionRef) -> Option<RawFd> {
        let state = self.slots[conn.conn_id as usize].state.lock().unwrap();
        if state.matches(conn.generation) && !state.retired {
            Some(state.fd)
        } else {
            None
        }
    }
}

pub struct FlushSubmission {
    pub fd: RawFd,
    pub iovecs: *const libc::iovec,
    pub iov_count: u32,
}

// SAFETY: All access to the embedded iovec pointers is synchronized through the per-slot
// mutex. The pointers only reference data owned by `inflight`, and those buffers remain
// stable until the writer consumes the corresponding CQE and updates the slot.
unsafe impl Send for ConnectionWriteState {}

pub enum WriteResult {
    Completed,
    ReadyAgain,
    NeedsResubmit,
    Error(i32),
    Stale,
    Idle,
}

pub fn encode_user_data(conn: ConnectionRef) -> u64 {
    ((conn.generation as u64) << 16) | conn.conn_id as u64
}

pub fn decode_user_data(user_data: u64) -> ConnectionRef {
    ConnectionRef {
        conn_id: user_data as u16,
        generation: (user_data >> 16) as u32,
    }
}

impl Default for ConnectionRegistry {
    fn default() -> Self {
        Self::new(SLAB_CAPACITY)
    }
}

#[cfg(test)]
mod tests {
    use super::{ConnectionRegistry, WriteResult, decode_user_data, encode_user_data};
    use crate::pipeline::ready_queue::ConnectionRef;

    #[test]
    fn user_data_round_trip() {
        let conn = ConnectionRef {
            conn_id: 17,
            generation: 0xdead_beef,
        };
        assert_eq!(decode_user_data(encode_user_data(conn)), conn);
    }

    #[test]
    fn retires_connection_after_read_close_and_write_completion() {
        let registry = ConnectionRegistry::new(4);
        let generation = registry.open(1, 42);
        let conn = ConnectionRef {
            conn_id: 1,
            generation,
        };
        assert!(registry.enqueue_response(conn, 0, 123, &[1, 2, 3]));
        let flush = registry.take_flush(conn).expect("flush");
        assert_eq!(flush.fd, 42);
        registry.mark_read_closed(1, generation, 1);
        match registry.handle_write_result(conn, 3) {
            WriteResult::Completed => {}
            _ => panic!("expected completed"),
        }
        assert!(registry.is_retired(1, generation));
    }

    #[test]
    fn partial_write_requires_resubmit() {
        let registry = ConnectionRegistry::new(4);
        let generation = registry.open(2, 77);
        let conn = ConnectionRef {
            conn_id: 2,
            generation,
        };
        assert!(registry.enqueue_response(conn, 0, 1, &[1, 2, 3, 4]));
        let _ = registry.take_flush(conn).expect("flush");
        match registry.handle_write_result(conn, 2) {
            WriteResult::NeedsResubmit => {}
            other => panic!(
                "expected NeedsResubmit, got {:?}",
                std::mem::discriminant(&other)
            ),
        }
        let _ = registry.take_flush(conn).expect("resubmit");
        match registry.handle_write_result(conn, 2) {
            WriteResult::Completed => {}
            _ => panic!("expected completed"),
        }
    }

    #[test]
    fn completion_of_inflight_write_requeues_later_response() {
        let registry = ConnectionRegistry::new(4);
        let generation = registry.open(3, 88);
        let conn = ConnectionRef {
            conn_id: 3,
            generation,
        };

        assert!(registry.enqueue_response(conn, 0, 1, &[1, 2, 3, 4]));
        let _ = registry.take_flush(conn).expect("initial flush");

        // A later response arrives while the first write is still in-flight.
        assert!(registry.enqueue_response(conn, 1, 2, &[5, 6, 7, 8]));

        match registry.handle_write_result(conn, 4) {
            WriteResult::ReadyAgain => {}
            other => panic!(
                "expected ReadyAgain, got {:?}",
                std::mem::discriminant(&other)
            ),
        }

        let _ = registry.take_flush(conn).expect("flush queued response");
        match registry.handle_write_result(conn, 4) {
            WriteResult::Completed => {}
            other => panic!(
                "expected Completed, got {:?}",
                std::mem::discriminant(&other)
            ),
        }
    }

    #[test]
    fn stale_generation_is_rejected() {
        let registry = ConnectionRegistry::new(2);
        let generation = registry.open(0, 10);
        let conn = ConnectionRef {
            conn_id: 0,
            generation,
        };
        registry.mark_read_closed(0, generation, 0);
        let next_generation = registry.open(0, 11);
        assert_ne!(generation, next_generation);
        assert!(!registry.enqueue_response(conn, 0, 0, &[1]));
    }
}
