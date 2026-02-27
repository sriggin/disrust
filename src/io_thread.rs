use std::io;
use std::os::unix::io::{AsRawFd, RawFd};
use std::ptr;

use disruptor::{Polling, SingleConsumerBarrier, SingleProducer};
use io_uring::{opcode, squeue::Entry, types::Fd};
use libc::iovec;
use slab::Slab;

use crate::buffer_pool::BufferPool;
use crate::config::{READ_BUF_SIZE, SLAB_CAPACITY};
use crate::metrics;
use crate::request_flow;
use crate::response_queue::RespPoller;
use crate::ring_types::{InferenceEvent, ResultStorage};

/// Encode operation type + connection key into io_uring user_data.
const OP_ACCEPT: u64 = 0;
const OP_READ: u64 = 1;
const OP_WRITE: u64 = 2;
const OP_EVENTFD: u64 = 3;

fn encode_user_data(op: u64, key: u16) -> u64 {
    (op << 32) | key as u64
}

fn decode_user_data(user_data: u64) -> (u64, u16) {
    (user_data >> 32, user_data as u16)
}

/// Thin zero-cost wrapper around `IoUring` that centralises submission helpers
/// and exposes a stable fd handle for future cross-thread `MSG_RING` posting.
struct IoUring {
    inner: io_uring::IoUring,
}

impl IoUring {
    fn new(entries: u32) -> io::Result<Self> {
        Ok(Self {
            inner: io_uring::IoUring::new(entries)?,
        })
    }

    /// The underlying ring fd — for future MSG_RING cross-thread posting.
    fn fd(&self) -> RawFd {
        self.inner.as_raw_fd()
    }

    /// Push an SQE, flushing the submission queue to the kernel if full.
    fn push(&mut self, sqe: &Entry) {
        loop {
            match unsafe { self.inner.submission().push(sqe) } {
                Ok(()) => return,
                Err(_) => {
                    // SQ full — flush to kernel and retry.
                    self.inner.submit().expect("submit failed during SQ flush");
                }
            }
        }
    }

    /// Block until at least `n` completions are available.
    fn wait(&mut self, n: usize) {
        self.inner.submit_and_wait(n).expect("submit_and_wait failed");
    }

    /// Drain all pending completions into a `(user_data, result)` vec.
    /// Collects eagerly so the borrow on the completion queue is released
    /// before any SQE submissions happen in the same loop iteration.
    fn drain_cqes(&mut self) -> Vec<(u64, i32)> {
        self.inner
            .completion()
            .map(|cqe| (cqe.user_data(), cqe.result()))
            .collect()
    }
}

struct Connection {
    fd: RawFd,
    read_buf: Box<[u8; READ_BUF_SIZE]>,
    read_len: usize,
    /// 1-byte header per response (num_vectors). Filled before building iovecs.
    write_headers: Vec<u8>,
    /// Raw f32 bytes (LE) for all responses. One bulk copy per response.
    write_payloads: Vec<u8>,
    write_segments: Vec<(usize, usize, usize)>,
    /// Scatter-gather list for Writev: [header_i, payload_i] per response. Built after headers/payloads are filled.
    pending_iovecs: Vec<iovec>,
    next_request_seq: u64,
    read_inflight: bool,
    write_inflight: bool,
}

impl Connection {
    fn new(fd: RawFd) -> Self {
        Self {
            fd,
            read_buf: Box::new([0u8; READ_BUF_SIZE]),
            read_len: 0,
            write_headers: Vec::with_capacity(256),
            write_payloads: Vec::with_capacity(4096),
            write_segments: Vec::with_capacity(128),
            pending_iovecs: Vec::with_capacity(512),
            next_request_seq: 0,
            read_inflight: false,
            write_inflight: false,
        }
    }

    /// Pointer and length for the unfilled tail of the read buffer.
    /// Safety: read_len is always <= READ_BUF_SIZE by construction.
    fn read_buf_tail(&mut self) -> (*mut u8, u32) {
        (
            unsafe { self.read_buf.as_mut_ptr().add(self.read_len) },
            (READ_BUF_SIZE - self.read_len) as u32,
        )
    }

    /// Rebuild pending_iovecs from write_headers/write_payloads/write_segments.
    /// Must only be called when write_inflight is false (buffers must not reallocate while in flight).
    fn build_iovecs(&mut self) {
        self.pending_iovecs.clear();
        let hdr = self.write_headers.as_ptr();
        let pay = self.write_payloads.as_ptr();
        for &(ho, ps, pl) in &self.write_segments {
            self.pending_iovecs.push(iovec {
                iov_base: unsafe { hdr.add(ho) as *mut libc::c_void },
                iov_len: 1,
            });
            self.pending_iovecs.push(iovec {
                iov_base: unsafe { pay.add(ps) as *mut libc::c_void },
                iov_len: pl,
            });
        }
    }
}

impl Drop for Connection {
    fn drop(&mut self) {
        // Return value intentionally ignored: on Linux, close() after EINTR still
        // closes the fd (retrying causes double-close); EIO means flush failed but
        // the fd is gone. Neither case is recoverable or worth panicking over.
        unsafe { libc::close(self.fd); }
    }
}

/// Reinterpret a slice of f32 as raw little-endian bytes.
/// Safety: f32 has no padding; any bit pattern is valid; alignment of u8 <= f32.
fn f32_slice_as_bytes(slice: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len() * 4) }
}

pub struct IoThread {
    pub thread_id: u8,
    pub listen_fd: RawFd,
    pub producer: SingleProducer<InferenceEvent, SingleConsumerBarrier>,
    pub response_poller: RespPoller,
    pub eventfd: RawFd,
    pub buffer_pool: &'static BufferPool,
}

impl IoThread {
    pub fn run(mut self) {
        let mut ring = IoUring::new(4096).expect("failed to create io_uring");
        let mut conns: Slab<Connection> = Slab::with_capacity(SLAB_CAPACITY);
        let mut eventfd_buf: u64 = 0;

        submit_accept(&mut ring, self.listen_fd);
        submit_eventfd_read(&mut ring, self.eventfd, &mut eventfd_buf);

        loop {
            ring.wait(1);

            for (user_data, result) in ring.drain_cqes() {
                let (op, key) = decode_user_data(user_data);
                match op {
                    OP_ACCEPT => self.handle_accept(&mut ring, &mut conns, result),
                    OP_READ => self.handle_read(&mut ring, &mut conns, key, result),
                    OP_WRITE => self.handle_write(&mut ring, &mut conns, key, result),
                    OP_EVENTFD => self.handle_eventfd(&mut ring, &mut conns, &mut eventfd_buf),
                    _ => {}
                }
            }
        }
    }

    fn handle_accept(&mut self, ring: &mut IoUring, conns: &mut Slab<Connection>, result: i32) {
        if result >= 0 {
            let client_fd = result as RawFd;
            let entry = conns.vacant_entry();
            let key = entry.key();
            entry.insert(Connection::new(client_fd));
            // key is always < SLAB_CAPACITY, which is asserted to fit in u16
            submit_read(ring, conns, key as u16);
        }
        submit_accept(ring, self.listen_fd);
    }

    fn handle_read(
        &mut self,
        ring: &mut IoUring,
        conns: &mut Slab<Connection>,
        key: u16,
        result: i32,
    ) {
        let key_usize = key as usize;
        if result <= 0 {
            conns.try_remove(key_usize);
            return;
        }

        let bytes_read = result as usize;
        let conn = &mut conns[key_usize];
        conn.read_inflight = false;
        conn.read_len += bytes_read;

        let buf = &conn.read_buf[..conn.read_len];
        match request_flow::process_requests_from_buffer(
            buf,
            &mut self.producer,
            self.buffer_pool,
            key,
            self.thread_id,
            &mut conn.next_request_seq,
        ) {
            Ok((consumed, num_published)) => {
                for _ in 0..num_published {
                    metrics::inc_req_occ();
                    metrics::inc_requests_published();
                }
                if consumed > 0 {
                    conn.read_buf.copy_within(consumed..conn.read_len, 0);
                    conn.read_len -= consumed;
                }
            }
            Err(e) => {
                eprintln!(
                    "io-{}: request flow error ({:?}), closing conn {}",
                    self.thread_id, e, key
                );
                conns.remove(key_usize);
                return;
            }
        }

        submit_read(ring, conns, key);
    }

    fn handle_write(
        &mut self,
        _ring: &mut IoUring,
        conns: &mut Slab<Connection>,
        key: u16,
        result: i32,
    ) {
        let key_usize = key as usize;
        if result < 0 {
            conns.try_remove(key_usize);
            return;
        }

        let conn = &mut conns[key_usize];
        conn.write_inflight = false;
        conn.write_headers.clear();
        conn.write_payloads.clear();
        conn.write_segments.clear();
        conn.pending_iovecs.clear();

        // If more data was queued while write was in flight, it's in the same buffers; we only
        // submit one write per drain, so no resubmit here.
    }

    fn handle_eventfd(
        &mut self,
        ring: &mut IoUring,
        conns: &mut Slab<Connection>,
        eventfd_buf: &mut u64,
    ) {
        match self.response_poller.poll() {
            Ok(mut guard) => {
                let mut write_keys: Vec<u16> = Vec::new();
                for resp in &mut guard {
                    if let Some(conn) = conns.get_mut(resp.conn_id as usize) {
                        let header_off = conn.write_headers.len();
                        conn.write_headers.push(resp.num_vectors);
                        let payload_start = conn.write_payloads.len();
                        let results = resp.results_slice();
                        let payload_len = results.len() * 4;
                        conn.write_payloads.extend_from_slice(f32_slice_as_bytes(results));
                        conn.write_segments
                            .push((header_off, payload_start, payload_len));
                        if !conn.write_inflight {
                            write_keys.push(resp.conn_id);
                        }
                    }
                    if let ResultStorage::Pooled(slice) = &resp.results {
                        slice.release();
                    }
                    metrics::dec_resp_occ();
                    metrics::inc_responses_sent();
                }
                write_keys.dedup();
                for key in write_keys {
                    if let Some(conn) = conns.get_mut(key as usize) {
                        conn.build_iovecs();
                    }
                    submit_write(ring, conns, key);
                }
            }
            Err(Polling::NoEvents) => {}
            Err(Polling::Shutdown) => return,
        }

        submit_eventfd_read(ring, self.eventfd, eventfd_buf);
    }
}

fn submit_accept(ring: &mut IoUring, listen_fd: RawFd) {
    let sqe = opcode::Accept::new(Fd(listen_fd), ptr::null_mut(), ptr::null_mut())
        .build()
        .user_data(encode_user_data(OP_ACCEPT, 0));
    ring.push(&sqe);
}

fn submit_read(ring: &mut IoUring, conns: &mut Slab<Connection>, key: u16) {
    let conn = &mut conns[key as usize];
    if conn.read_inflight {
        return;
    }
    conn.read_inflight = true;
    let (buf_ptr, buf_len) = conn.read_buf_tail();

    let sqe = opcode::Read::new(Fd(conn.fd), buf_ptr, buf_len)
        .build()
        .user_data(encode_user_data(OP_READ, key));
    ring.push(&sqe);
}

fn submit_write(ring: &mut IoUring, conns: &mut Slab<Connection>, key: u16) {
    let conn = &mut conns[key as usize];
    if conn.write_inflight || conn.pending_iovecs.is_empty() {
        return;
    }
    conn.write_inflight = true;

    let iovecs_ptr = conn.pending_iovecs.as_ptr();
    let iovecs_len = conn.pending_iovecs.len() as u32;

    let sqe = opcode::Writev::new(Fd(conn.fd), iovecs_ptr, iovecs_len)
        .build()
        .user_data(encode_user_data(OP_WRITE, key));
    ring.push(&sqe);
}

fn submit_eventfd_read(ring: &mut IoUring, eventfd: RawFd, buf: &mut u64) {
    // buf as *mut u64 as *mut u8: io_uring Read requires a *mut u8 buffer;
    // the eventfd kernel ABI always writes exactly 8 bytes (a u64 counter value).
    let sqe = opcode::Read::new(Fd(eventfd), buf as *mut u64 as *mut u8, 8)
        .build()
        .user_data(encode_user_data(OP_EVENTFD, 0));
    ring.push(&sqe);
}
