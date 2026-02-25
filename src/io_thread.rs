use std::os::unix::io::RawFd;
use std::ptr;

use disruptor::{Polling, Producer, RingBufferFull, SingleConsumerBarrier, SingleProducer};
use io_uring::{IoUring, opcode, squeue::Entry, types::Fd};
use slab::Slab;

use crate::buffer_pool::BufferPool;
use crate::constants::FEATURE_DIM;
use crate::metrics;
use crate::protocol;
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

const READ_BUF_SIZE: usize = 65536;

/// Max concurrent connections per IO thread. Must fit in u16 (conn_id).
const SLAB_CAPACITY: usize = 4096;
const _: () = assert!(SLAB_CAPACITY <= u16::MAX as usize, "conn_id is u16");

struct Connection {
    fd: RawFd,
    read_buf: Box<[u8; READ_BUF_SIZE]>,
    read_len: usize,
    write_buf: Vec<u8>,
    write_pos: usize,
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
            write_buf: Vec::with_capacity(4096),
            write_pos: 0,
            next_request_seq: 0,
            read_inflight: false,
            write_inflight: false,
        }
    }
}

/// Push an SQE, flushing the submission queue if full.
fn push_sqe(ring: &mut IoUring, sqe: &Entry) {
    loop {
        let result = unsafe { ring.submission().push(sqe) };
        match result {
            Ok(()) => return,
            Err(_) => {
                // SQ full, flush pending submissions to kernel and retry
                ring.submit().expect("submit failed during SQ flush");
            }
        }
    }
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

        // Submit initial accept
        submit_accept(&mut ring, self.listen_fd);
        // Submit eventfd read to wake us when responses arrive
        submit_eventfd_read(&mut ring, self.eventfd, &mut eventfd_buf);

        loop {
            ring.submit_and_wait(1).expect("submit_and_wait failed");

            let cqes: Vec<(u64, i32)> = ring
                .completion()
                .map(|cqe| (cqe.user_data(), cqe.result()))
                .collect();

            for (user_data, result) in cqes {
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
            if let Some(conn) = conns.try_remove(key_usize) {
                unsafe {
                    libc::close(conn.fd);
                }
            }
            return;
        }

        let bytes_read = result as usize;
        let conn = &mut conns[key_usize];
        conn.read_inflight = false;
        conn.read_len += bytes_read;

        // Parse as many complete requests as possible
        let mut consumed = 0;
        while consumed < conn.read_len {
            let buf = &conn.read_buf[consumed..conn.read_len];
            match protocol::try_parse_request(buf) {
                protocol::ParseResult::Complete {
                    num_vectors,
                    bytes_consumed,
                } => {
                    let feature_bytes = &buf[4..bytes_consumed];
                    let seq = conn.next_request_seq;
                    conn.next_request_seq += 1;
                    let thread_id = self.thread_id;
                    let conn_id = key;

                    // Allocate from pool and copy features
                    let feature_count = num_vectors as usize * FEATURE_DIM;
                    let mut pool_slice = match self.buffer_pool.alloc(feature_count) {
                        Ok(slice) => slice,
                        Err(e) => {
                            // Request too large (bad client) - close connection
                            eprintln!(
                                "io-{}: buffer pool allocation failed ({:?}), closing conn {}",
                                thread_id, e, conn_id
                            );
                            let conn = conns.remove(key_usize);
                            unsafe {
                                libc::close(conn.fd);
                            }
                            return;
                        }
                    };

                    protocol::copy_features(feature_bytes, pool_slice.as_mut_slice(), num_vectors);
                    let mut features = Some(pool_slice.freeze());

                    loop {
                        match self.producer.try_publish(|slot| {
                            slot.io_thread_id = thread_id;
                            slot.conn_id = conn_id;
                            slot.request_seq = seq;
                            slot.num_vectors = num_vectors;
                            slot.features =
                                features.take().expect("features already moved into ring");
                        }) {
                            Ok(_) => {
                                metrics::inc_req_occ();
                                metrics::inc_requests_published();
                                break;
                            }
                            Err(RingBufferFull) => {
                                metrics::inc_req_ring_full();
                                std::hint::spin_loop();
                            }
                        }
                    }

                    consumed += bytes_consumed;
                }
                protocol::ParseResult::Incomplete(_) => break,
                protocol::ParseResult::Error(_) => {
                    let conn = conns.remove(key_usize);
                    unsafe {
                        libc::close(conn.fd);
                    }
                    return;
                }
            }
        }

        // Compact read buffer
        let conn = &mut conns[key_usize];
        if consumed > 0 {
            conn.read_buf.copy_within(consumed..conn.read_len, 0);
            conn.read_len -= consumed;
        }

        submit_read(ring, conns, key);
    }

    fn handle_write(
        &mut self,
        ring: &mut IoUring,
        conns: &mut Slab<Connection>,
        key: u16,
        result: i32,
    ) {
        let key_usize = key as usize;
        if result < 0 {
            if let Some(conn) = conns.try_remove(key_usize) {
                unsafe {
                    libc::close(conn.fd);
                }
            }
            return;
        }

        let bytes_written = result as usize;
        let conn = &mut conns[key_usize];
        conn.write_inflight = false;
        conn.write_pos += bytes_written;

        if conn.write_pos >= conn.write_buf.len() {
            conn.write_buf.clear();
            conn.write_pos = 0;
        } else {
            submit_write(ring, conns, key);
        }
    }

    fn handle_eventfd(
        &mut self,
        ring: &mut IoUring,
        conns: &mut Slab<Connection>,
        eventfd_buf: &mut u64,
    ) {
        // Drain all available responses from the disruptor
        match self.response_poller.poll() {
            Ok(mut guard) => {
                // Append all responses BEFORE submitting any writes.
                // write_buf may reallocate as data is appended; capturing buf_ptr
                // (inside submit_write) before all appends are done would leave a
                // dangling pointer in the SQE once the Vec moves its allocation.
                let mut write_keys: Vec<u16> = Vec::new();
                for resp in &mut guard {
                    if let Some(conn) = conns.get_mut(resp.conn_id as usize) {
                        protocol::write_response(
                            &mut conn.write_buf,
                            resp.num_vectors,
                            resp.results_slice(),
                        );
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
                // Responses arrive ordered per connection, so dedup collapses runs.
                write_keys.dedup();
                for key in write_keys {
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
    push_sqe(ring, &sqe);
}

fn submit_read(ring: &mut IoUring, conns: &mut Slab<Connection>, key: u16) {
    let conn = &mut conns[key as usize];
    if conn.read_inflight {
        return;
    }
    conn.read_inflight = true;

    let buf_ptr = unsafe { conn.read_buf.as_mut_ptr().add(conn.read_len) };
    let buf_len = (READ_BUF_SIZE - conn.read_len) as u32;

    let sqe = opcode::Read::new(Fd(conn.fd), buf_ptr, buf_len)
        .build()
        .user_data(encode_user_data(OP_READ, key));
    push_sqe(ring, &sqe);
}

fn submit_write(ring: &mut IoUring, conns: &mut Slab<Connection>, key: u16) {
    let conn = &mut conns[key as usize];
    if conn.write_inflight {
        return;
    }
    conn.write_inflight = true;

    let buf_ptr = unsafe { conn.write_buf.as_ptr().add(conn.write_pos) };
    let buf_len = (conn.write_buf.len() - conn.write_pos) as u32;

    let sqe = opcode::Write::new(Fd(conn.fd), buf_ptr, buf_len)
        .build()
        .user_data(encode_user_data(OP_WRITE, key));
    push_sqe(ring, &sqe);
}

fn submit_eventfd_read(ring: &mut IoUring, eventfd: RawFd, buf: &mut u64) {
    let sqe = opcode::Read::new(Fd(eventfd), buf as *mut u64 as *mut u8, 8)
        .build()
        .user_data(encode_user_data(OP_EVENTFD, 0));
    push_sqe(ring, &sqe);
}
