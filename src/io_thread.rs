use std::io;
use std::os::unix::io::RawFd;
use std::ptr;

use std::collections::VecDeque;

use disruptor::{Polling, SingleConsumerBarrier, SingleProducer};
use io_uring::{opcode, squeue::Entry, types::Fd};
use slab::Slab;
use spsc_bip_buffer::bip_buffer_with_len;

use crate::buffer_pool::BufferPool;
use crate::config::{READ_BUF_SIZE, SLAB_CAPACITY, WRITE_BIP_CAPACITY};
use crate::metrics;
use crate::protocol;
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

/// Max in-flight Write SQEs per connection. Allows pipelining without spinning.
const MAX_WRITES_IN_FLIGHT_PER_CONN: usize = 4;

/// Wraps the bip reader so we can have multiple writes in flight: we track submitted-but-not-yet-
/// consumed lengths and expose only the "not yet submitted" suffix of valid().
struct BipWriteState {
    reader: spsc_bip_buffer::BipBufferReader,
    in_flight_lens: VecDeque<usize>,
}

impl BipWriteState {
    fn new(reader: spsc_bip_buffer::BipBufferReader) -> Self {
        Self {
            reader,
            in_flight_lens: VecDeque::new(),
        }
    }

    /// Bytes available to submit (excluding already-in-flight).
    fn valid_available(&mut self) -> &[u8] {
        let skip: usize = self.in_flight_lens.iter().sum();
        let v = self.reader.valid();
        if skip >= v.len() {
            return &[];
        }
        &v[skip..]
    }

    fn submit(&mut self, len: usize) {
        self.in_flight_lens.push_back(len);
    }

    /// Call when a write CQE completes.
    /// `written` is the CQE result (actual bytes written by the kernel).
    /// Consumes exactly `written` bytes from the bip buffer. If the kernel wrote fewer
    /// than submitted (partial write), the remainder is re-queued at the front so
    /// submit_write resubmits it on the next iteration.
    fn on_complete(&mut self, written: usize) {
        let submitted = self
            .in_flight_lens
            .pop_front()
            .expect("in-flight queue empty");
        self.reader.consume(written);
        let remaining = submitted - written;
        if remaining > 0 {
            // Partial write: re-queue remainder at front for immediate resubmission.
            self.in_flight_lens.push_front(remaining);
        }
    }

    fn in_flight_count(&self) -> usize {
        self.in_flight_lens.len()
    }
}

/// Thin zero-cost wrapper around `IoUring` that centralises submission helpers.
struct IoUring {
    inner: io_uring::IoUring,
}

impl IoUring {
    fn new(entries: u32) -> io::Result<Self> {
        Ok(Self {
            inner: io_uring::IoUring::new(entries)?,
        })
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
        self.inner
            .submit_and_wait(n)
            .expect("submit_and_wait failed");
    }

    /// Drain all pending completions into the provided buffer.
    /// Collects eagerly so the borrow on the completion queue is released
    /// before any SQE submissions happen in the same loop iteration.
    fn drain_cqes_into(&mut self, buf: &mut Vec<(u64, i32)>) {
        buf.extend(
            self.inner
                .completion()
                .map(|cqe| (cqe.user_data(), cqe.result())),
        );
    }
}

struct Connection {
    fd: RawFd,
    read_buf: Box<[u8; READ_BUF_SIZE]>,
    read_len: usize,
    write_bip_writer: spsc_bip_buffer::BipBufferWriter,
    write_bip: BipWriteState,
    next_request_seq: u64,
    read_inflight: bool,
}

impl Connection {
    fn new(fd: RawFd) -> Self {
        let (write_bip_writer, write_bip_reader) = bip_buffer_with_len(WRITE_BIP_CAPACITY);
        Self {
            fd,
            read_buf: Box::new([0u8; READ_BUF_SIZE]),
            read_len: 0,
            write_bip_writer,
            write_bip: BipWriteState::new(write_bip_reader),
            next_request_seq: 0,
            read_inflight: false,
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
}

impl Drop for Connection {
    fn drop(&mut self) {
        // Return value intentionally ignored: on Linux, close() after EINTR still
        // closes the fd (retrying causes double-close); EIO means flush failed but
        // the fd is gone. Neither case is recoverable or worth panicking over.
        unsafe {
            libc::close(self.fd);
        }
    }
}

/// Reinterpret a slice of f32 as raw little-endian bytes.
/// Safety: f32 has no padding; any bit pattern is valid; alignment of u8 <= f32.
fn f32_slice_as_bytes(slice: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len() * 4) }
}

pub struct IoThread {
    thread_id: u8,
    listen_fd: RawFd,
    producer: SingleProducer<InferenceEvent, SingleConsumerBarrier>,
    response_poller: RespPoller,
    eventfd: RawFd,
    buffer_pool: &'static BufferPool,
}

impl IoThread {
    pub fn new(
        thread_id: u8,
        listen_fd: RawFd,
        producer: SingleProducer<InferenceEvent, SingleConsumerBarrier>,
        response_poller: RespPoller,
        eventfd: RawFd,
        buffer_pool: &'static BufferPool,
    ) -> Self {
        Self {
            thread_id,
            listen_fd,
            producer,
            response_poller,
            eventfd,
            buffer_pool,
        }
    }

    pub fn run(self) {
        let mut state = RunState {
            ring: IoUring::new(4096).expect("failed to create io_uring"),
            conns: Slab::with_capacity(SLAB_CAPACITY),
            eventfd_buf: 0,
            listen_fd: self.listen_fd,
            producer: self.producer,
            buffer_pool: self.buffer_pool,
            thread_id: self.thread_id,
            eventfd: self.eventfd,
            write_keys: Vec::new(),
            inner_cqes: Vec::new(),
        };
        let mut response_poller = self.response_poller;

        submit_accept(&mut state.ring, state.listen_fd);
        submit_eventfd_read(&mut state.ring, state.eventfd, &mut state.eventfd_buf);

        let mut cqe_buf: Vec<(u64, i32)> = Vec::new();
        loop {
            state.ring.wait(1);
            cqe_buf.clear();
            state.ring.drain_cqes_into(&mut cqe_buf);
            state.process_completions(&mut response_poller, &cqe_buf);
        }
    }
}

/// All mutable state owned by a running IO thread.
struct RunState {
    ring: IoUring,
    conns: Slab<Connection>,
    eventfd_buf: u64,
    listen_fd: RawFd,
    producer: SingleProducer<InferenceEvent, SingleConsumerBarrier>,
    buffer_pool: &'static BufferPool,
    thread_id: u8,
    eventfd: RawFd,
    /// Reusable buffer: connection keys that need a write submitted after an OP_EVENTFD batch.
    write_keys: Vec<u16>,
    /// Reusable buffer: CQEs drained inside the bip-full retry loop.
    inner_cqes: Vec<(u64, i32)>,
}

impl RunState {
    /// Process one batch of CQEs.
    fn process_completions(&mut self, response_poller: &mut RespPoller, cqes: &[(u64, i32)]) {
        for &(user_data, result) in cqes {
            let (op, key) = decode_user_data(user_data);
            match op {
                OP_ACCEPT => self.handle_accept(result),
                OP_READ => self.handle_read(key, result),
                OP_WRITE => self.handle_write(key, result),
                OP_EVENTFD => {
                    match response_poller.poll() {
                        Ok(mut guard) => {
                            self.write_keys.clear();
                            for resp in &mut guard {
                                let conn_id = resp.conn_id;
                                let num_vectors = resp.num_vectors;
                                let results = resp.results_slice();
                                let len = protocol::response_size(results.len());
                                metrics::dec_resp_occ();

                                // Skip silently if the connection was closed before
                                // we got to write the response.
                                if self.conns.contains(conn_id as usize) {
                                    loop {
                                        // Connection may have been removed by a write/read
                                        // error handled during the wait below; if so, stop.
                                        let Some(conn) = self.conns.get_mut(conn_id as usize)
                                        else {
                                            break;
                                        };
                                        if let Some(mut r) = conn.write_bip_writer.reserve(len) {
                                            r[0] = num_vectors;
                                            r[1..].copy_from_slice(f32_slice_as_bytes(results));
                                            r.send();
                                            if conn.write_bip.in_flight_count()
                                                < MAX_WRITES_IN_FLIGHT_PER_CONN
                                                && !conn.write_bip.valid_available().is_empty()
                                            {
                                                self.write_keys.push(conn_id);
                                            }
                                            metrics::inc_responses_sent();
                                            break;
                                        }
                                        // Bip full: flush pending writes for this connection
                                        // before waiting so OP_WRITE CQEs will arrive.
                                        // Without this, no writes would be in-kernel and
                                        // ring.wait(1) would block indefinitely.
                                        submit_write(&mut self.ring, &mut self.conns, conn_id);
                                        // Dispatch write/read/accept CQEs directly — we must not
                                        // poll the disruptor again while holding the guard.
                                        self.ring.wait(1);
                                        self.inner_cqes.clear();
                                        self.ring.drain_cqes_into(&mut self.inner_cqes);
                                        for i in 0..self.inner_cqes.len() {
                                            let (ud, res) = self.inner_cqes[i];
                                            let (iop, ikey) = decode_user_data(ud);
                                            match iop {
                                                OP_ACCEPT => self.handle_accept(res),
                                                OP_READ => self.handle_read(ikey, res),
                                                OP_WRITE => self.handle_write(ikey, res),
                                                OP_EVENTFD => submit_eventfd_read(
                                                    &mut self.ring,
                                                    self.eventfd,
                                                    &mut self.eventfd_buf,
                                                ),
                                                _ => {}
                                            }
                                        }
                                    }
                                }
                                // Release result pool slice in all outcomes: written into bip,
                                // conn gone before write, or conn gone during wait.
                                if let ResultStorage::Pooled(slice) = &resp.results {
                                    slice.release();
                                }
                            }
                            self.write_keys.sort_unstable();
                            self.write_keys.dedup();
                            for i in 0..self.write_keys.len() {
                                let key = self.write_keys[i];
                                if self.conns.contains(key as usize) {
                                    submit_write(&mut self.ring, &mut self.conns, key);
                                }
                            }

                            // Retry any connections stalled by RingBufferFull.
                            //
                            // When the disruptor ring was full during a previous handle_read,
                            // try_publish returned RingBufferFull immediately (without spinning).
                            // If the read buffer was also full at that moment, no new OP_READ
                            // was submitted, leaving the connection with unprocessed bytes but
                            // no pending CQE to wake it up. Now that the ring has drained (we
                            // just received responses), re-parse those buffers.
                            self.write_keys.clear();
                            for (k, c) in self.conns.iter() {
                                if !c.read_inflight && c.read_len > 0 {
                                    self.write_keys.push(k as u16);
                                }
                            }
                            for i in 0..self.write_keys.len() {
                                let key = self.write_keys[i];
                                self.parse_and_maybe_read(key);
                            }
                        }
                        Err(Polling::NoEvents) => {}
                        Err(Polling::Shutdown) => return,
                    }
                    submit_eventfd_read(&mut self.ring, self.eventfd, &mut self.eventfd_buf);
                }
                _ => {}
            }
        }
    }

    fn handle_accept(&mut self, result: i32) {
        if result >= 0 {
            let client_fd = result as RawFd;
            let entry = self.conns.vacant_entry();
            let key = entry.key();
            entry.insert(Connection::new(client_fd));
            submit_read(&mut self.ring, &mut self.conns, key as u16);
        }
        submit_accept(&mut self.ring, self.listen_fd);
    }

    fn handle_read(&mut self, key: u16, result: i32) {
        let key_usize = key as usize;
        if result <= 0 {
            self.conns.try_remove(key_usize);
            return;
        }

        let bytes_read = result as usize;
        let Some(conn) = self.conns.get_mut(key_usize) else {
            return;
        };
        conn.read_inflight = false;
        conn.read_len += bytes_read;

        self.parse_and_maybe_read(key);
    }

    /// Parse the connection's existing read buffer and resubmit a read if space is available.
    ///
    /// Called both from `handle_read` (after new bytes arrive) and from the `OP_EVENTFD`
    /// handler (to retry connections stalled by `RingBufferFull` with a full buffer).
    fn parse_and_maybe_read(&mut self, key: u16) {
        let key_usize = key as usize;
        let Some(conn) = self.conns.get_mut(key_usize) else {
            return;
        };
        let buf = &conn.read_buf[..conn.read_len];
        let fd = conn.fd;
        match request_flow::process_requests_from_buffer(
            buf,
            &mut self.producer,
            self.buffer_pool,
            key,
            fd,
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
                self.conns.remove(key_usize);
                return;
            }
        }

        // Resubmit a read if there is room in the buffer.
        // If read_len == READ_BUF_SIZE, either the ring was full (RingBufferFull broke
        // out of the parse loop with consumed == 0) or pool exhaustion prevented alloc.
        // In the ring-full case the OP_EVENTFD retry below will re-enter here once the
        // ring has drained; in the pool-exhaustion case the closure spins until space
        // is available, so this branch is only hit when the buffer is genuinely full.
        let c = &self.conns[key as usize];
        if c.read_len < READ_BUF_SIZE {
            submit_read(&mut self.ring, &mut self.conns, key);
        }
    }

    fn handle_write(&mut self, key: u16, result: i32) {
        let key_usize = key as usize;
        if result < 0 {
            self.conns.try_remove(key_usize);
            return;
        }
        let Some(conn) = self.conns.get_mut(key_usize) else {
            return;
        };
        conn.write_bip.on_complete(result as usize);
        submit_write(&mut self.ring, &mut self.conns, key);
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
    while conn.write_bip.in_flight_count() < MAX_WRITES_IN_FLIGHT_PER_CONN {
        let available = conn.write_bip.valid_available();
        if available.is_empty() {
            break;
        }
        let chunk_len = available.len().min(u32::MAX as usize) as u32;
        let buf_ptr = available.as_ptr();

        conn.write_bip.submit(chunk_len as usize);

        let sqe = opcode::Write::new(Fd(conn.fd), buf_ptr, chunk_len)
            .build()
            .user_data(encode_user_data(OP_WRITE, key));
        ring.push(&sqe);
    }
}

fn submit_eventfd_read(ring: &mut IoUring, eventfd: RawFd, buf: &mut u64) {
    // buf as *mut u64 as *mut u8: io_uring Read requires a *mut u8 buffer;
    // the eventfd kernel ABI always writes exactly 8 bytes (a u64 counter value).
    let sqe = opcode::Read::new(Fd(eventfd), buf as *mut u64 as *mut u8, 8)
        .build()
        .user_data(encode_user_data(OP_EVENTFD, 0));
    ring.push(&sqe);
}
