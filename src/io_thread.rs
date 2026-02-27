use std::os::unix::io::RawFd;
use std::ptr;

use disruptor::{Polling, SingleConsumerBarrier, SingleProducer};
use io_uring::{IoUring, opcode, squeue::Entry, types::Fd};
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
                let conn = conns.remove(key_usize);
                unsafe {
                    libc::close(conn.fd);
                }
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
            if let Some(conn) = conns.try_remove(key_usize) {
                unsafe {
                    libc::close(conn.fd);
                }
            }
            return;
        }

        let _bytes_written = result as usize;
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
                        conn.write_payloads.extend_from_slice(unsafe {
                            std::slice::from_raw_parts(results.as_ptr() as *const u8, payload_len)
                        });
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
                        conn.pending_iovecs.clear();
                        let hdr = conn.write_headers.as_ptr();
                        let pay = conn.write_payloads.as_ptr();
                        for &(ho, ps, pl) in &conn.write_segments {
                            conn.pending_iovecs.push(iovec {
                                iov_base: unsafe { hdr.add(ho) as *mut libc::c_void },
                                iov_len: 1,
                            });
                            conn.pending_iovecs.push(iovec {
                                iov_base: unsafe { pay.add(ps) as *mut libc::c_void },
                                iov_len: pl,
                            });
                        }
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
    if conn.write_inflight || conn.pending_iovecs.is_empty() {
        return;
    }
    conn.write_inflight = true;

    let iovecs_ptr = conn.pending_iovecs.as_ptr();
    let iovecs_len = conn.pending_iovecs.len() as u32;

    let sqe = opcode::Writev::new(Fd(conn.fd), iovecs_ptr, iovecs_len)
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
