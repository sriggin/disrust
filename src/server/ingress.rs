//! Ingress-only IO thread for the ONNX/CUDA server pipeline.

use std::io;
use std::os::unix::io::RawFd;
use std::ptr;

use disruptor::{SingleConsumerBarrier, SingleProducer};
use io_uring::{opcode, squeue::Entry, types::Fd};
use slab::Slab;

use crate::buffer_pool::PoolAllocator;
use crate::config::{READ_BUF_SIZE, SLAB_CAPACITY};
use crate::metrics;
use crate::request_flow;
use crate::ring_types::InferenceEvent;

const OP_ACCEPT: u64 = 0;
const OP_READ: u64 = 1;

fn encode_user_data(op: u64, key: u16) -> u64 {
    (op << 32) | key as u64
}

fn decode_user_data(user_data: u64) -> (u64, u16) {
    (user_data >> 32, user_data as u16)
}

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

    fn wait(&mut self, n: usize) {
        self.inner
            .submit_and_wait(n)
            .expect("submit_and_wait failed");
    }

    fn drain_cqes_into(&mut self, buf: &mut Vec<(u64, i32)>) {
        buf.extend(self.inner.completion().map(|c| (c.user_data(), c.result())));
    }
}

struct Connection {
    fd: RawFd,
    read_buf: Box<[u8; READ_BUF_SIZE]>,
    read_len: usize,
    next_request_seq: u64,
    read_inflight: bool,
}

impl Connection {
    fn new(fd: RawFd) -> Self {
        Self {
            fd,
            read_buf: Box::new([0u8; READ_BUF_SIZE]),
            read_len: 0,
            next_request_seq: 0,
            read_inflight: false,
        }
    }

    fn read_buf_tail(&mut self) -> (*mut u8, u32) {
        (
            unsafe { self.read_buf.as_mut_ptr().add(self.read_len) },
            (READ_BUF_SIZE - self.read_len) as u32,
        )
    }
}

impl Drop for Connection {
    fn drop(&mut self) {
        unsafe {
            libc::close(self.fd);
        }
    }
}

pub struct IngressThread {
    thread_id: u8,
    listen_fd: RawFd,
    producer: SingleProducer<InferenceEvent, SingleConsumerBarrier>,
    allocator: PoolAllocator,
}

impl IngressThread {
    pub fn new(
        thread_id: u8,
        listen_fd: RawFd,
        producer: SingleProducer<InferenceEvent, SingleConsumerBarrier>,
        allocator: PoolAllocator,
    ) -> Self {
        Self {
            thread_id,
            listen_fd,
            producer,
            allocator,
        }
    }

    pub fn run(mut self) {
        let mut ring = IoUring::new(4096).expect("io_uring creation failed");
        let mut conns: Slab<Connection> = Slab::with_capacity(SLAB_CAPACITY);
        let mut cqe_buf: Vec<(u64, i32)> = Vec::new();
        let mut retry_keys: Vec<u16> = Vec::new();
        submit_accept(&mut ring, self.listen_fd);

        loop {
            retry_keys.clear();
            for (k, c) in conns.iter() {
                if !c.read_inflight && c.read_len > 0 {
                    retry_keys.push(k as u16);
                }
            }
            if !retry_keys.is_empty() {
                for &key in &retry_keys {
                    parse_and_maybe_read(
                        &mut ring,
                        &mut conns,
                        &mut self.producer,
                        &mut self.allocator,
                        self.thread_id,
                        key,
                    );
                }
                continue;
            }

            ring.wait(1);
            cqe_buf.clear();
            ring.drain_cqes_into(&mut cqe_buf);

            for &(user_data, result) in &cqe_buf {
                let (op, key) = decode_user_data(user_data);
                match op {
                    OP_ACCEPT => handle_accept(&mut ring, &mut conns, result, self.listen_fd),
                    OP_READ => handle_read(
                        &mut ring,
                        &mut conns,
                        &mut self.producer,
                        &mut self.allocator,
                        self.thread_id,
                        key,
                        result,
                    ),
                    _ => {}
                }
            }

            retry_keys.clear();
            for (k, c) in conns.iter() {
                if !c.read_inflight && c.read_len > 0 {
                    retry_keys.push(k as u16);
                }
            }
            for &key in &retry_keys {
                parse_and_maybe_read(
                    &mut ring,
                    &mut conns,
                    &mut self.producer,
                    &mut self.allocator,
                    self.thread_id,
                    key,
                );
            }
        }
    }
}

fn handle_accept(ring: &mut IoUring, conns: &mut Slab<Connection>, result: i32, listen_fd: RawFd) {
    if result >= 0 {
        let client_fd = result as RawFd;
        let entry = conns.vacant_entry();
        let key = entry.key();
        entry.insert(Connection::new(client_fd));
        submit_read(ring, conns, key as u16);
    }
    submit_accept(ring, listen_fd);
}

fn handle_read(
    ring: &mut IoUring,
    conns: &mut Slab<Connection>,
    producer: &mut SingleProducer<InferenceEvent, SingleConsumerBarrier>,
    allocator: &mut PoolAllocator,
    thread_id: u8,
    key: u16,
    result: i32,
) {
    let key_usize = key as usize;
    metrics::inc_read_cqes();
    if result <= 0 {
        if result < 0 {
            metrics::inc_read_negative();
        }
        conns.try_remove(key_usize);
        return;
    }
    let bytes_read = result as usize;
    metrics::add_read_bytes(bytes_read as u64);
    let Some(conn) = conns.get_mut(key_usize) else {
        return;
    };
    conn.read_inflight = false;
    conn.read_len += bytes_read;

    parse_and_maybe_read(ring, conns, producer, allocator, thread_id, key);
}

fn parse_and_maybe_read(
    ring: &mut IoUring,
    conns: &mut Slab<Connection>,
    producer: &mut SingleProducer<InferenceEvent, SingleConsumerBarrier>,
    allocator: &mut PoolAllocator,
    thread_id: u8,
    key: u16,
) {
    let key_usize = key as usize;
    let Some(conn) = conns.get_mut(key_usize) else {
        return;
    };
    let buf = &conn.read_buf[..conn.read_len];
    let fd = conn.fd;

    match request_flow::process_requests_from_buffer(
        buf,
        producer,
        allocator,
        key,
        fd,
        thread_id,
        &mut conn.next_request_seq,
    ) {
        Ok(outcome) => {
            if outcome.consumed > 0 {
                conn.read_buf
                    .copy_within(outcome.consumed..conn.read_len, 0);
                conn.read_len -= outcome.consumed;
            }
            metrics::add_bytes_consumed(outcome.consumed as u64);
            metrics::set_buffered_bytes(conn.read_len);
            if outcome.needs_read {
                submit_read(ring, conns, key);
            }
        }
        Err(e) => {
            eprintln!(
                "io-{}: request parse error ({:?}), closing conn {}",
                thread_id, e, key
            );
            conns.remove(key_usize);
            return;
        }
    }

    let c = &conns[key as usize];
    if c.read_len == 0 {
        submit_read(ring, conns, key);
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
    metrics::inc_read_submits();
}
