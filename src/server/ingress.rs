//! Shard-owned IO thread for the ONNX/CUDA server pipeline.

use std::collections::VecDeque;
use std::io;
use std::os::unix::io::RawFd;
use std::ptr;
use std::sync::{Arc, Mutex};

use disruptor::Producer;
use io_uring::{opcode, squeue::Entry, types::Fd};
use slab::Slab;

use crate::buffer_pool::PoolAllocator;
use crate::clock::elapsed_since_ns;
use crate::config::{READ_BUF_SIZE, SLAB_CAPACITY, WRITE_BUF_SIZE};
use crate::connection_id::ConnectionRef;
use crate::metrics;
use crate::pipeline::connection_registry::ConnectionRegistry;
use crate::pipeline::response_queue::ResponseQueue;
use crate::request_flow;
use crate::ring_types::InferenceEvent;

const OP_ACCEPT: u64 = 0;
const OP_READ: u64 = 1;
const OP_WRITE: u64 = 2;
const OP_NOTIFY: u64 = 3;
const MAX_IOVECS_PER_WRITE: usize = 64;

fn encode_user_data(op: u64, data: u32) -> u64 {
    (op << 32) | data as u64
}

fn decode_user_data(user_data: u64) -> (u64, u32) {
    (user_data >> 32, user_data as u32)
}

struct IoUring {
    inner: io_uring::IoUring,
    outstanding: usize,
}

impl IoUring {
    fn new(entries: u32) -> io::Result<Self> {
        Ok(Self {
            inner: io_uring::IoUring::new(entries)?,
            outstanding: 0,
        })
    }

    fn push(&mut self, sqe: &Entry) {
        loop {
            match unsafe { self.inner.submission().push(sqe) } {
                Ok(()) => {
                    self.outstanding += 1;
                    return;
                }
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

    fn submit(&mut self) {
        if self.outstanding > 0 {
            self.inner.submit().expect("io_uring submit failed");
        }
    }

    fn drain_cqes_into(&mut self, buf: &mut Vec<(u64, i32)>) {
        for cqe in self.inner.completion() {
            self.outstanding = self.outstanding.saturating_sub(1);
            buf.push((cqe.user_data(), cqe.result()));
        }
    }
}

struct ResponseFrame {
    published_at_ns: u64,
    len: usize,
    offset: usize,
    data: [u8; WRITE_BUF_SIZE],
}

impl ResponseFrame {
    fn new(published_at_ns: u64, bytes: &[u8]) -> Self {
        debug_assert!(bytes.len() <= WRITE_BUF_SIZE);
        let mut data = [0u8; WRITE_BUF_SIZE];
        data[..bytes.len()].copy_from_slice(bytes);
        Self {
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

struct Connection {
    fd: RawFd,
    conn: ConnectionRef,
    read_buf: Box<[u8; READ_BUF_SIZE]>,
    read_len: usize,
    next_request_seq: u64,
    read_inflight: bool,
    read_closed: bool,
    parse_queued: bool,
    write_closed: bool,
    write_inflight: bool,
    ready_queued: bool,
    queue: VecDeque<Box<ResponseFrame>>,
    inflight: VecDeque<Box<ResponseFrame>>,
    inflight_iovecs: [libc::iovec; MAX_IOVECS_PER_WRITE],
    inflight_iov_count: usize,
}

impl Connection {
    fn new(fd: RawFd, conn: ConnectionRef) -> Self {
        Self {
            fd,
            conn,
            read_buf: Box::new([0u8; READ_BUF_SIZE]),
            read_len: 0,
            next_request_seq: 0,
            read_inflight: false,
            read_closed: false,
            parse_queued: false,
            write_closed: false,
            write_inflight: false,
            ready_queued: false,
            queue: VecDeque::new(),
            inflight: VecDeque::new(),
            inflight_iovecs: [libc::iovec {
                iov_base: std::ptr::null_mut(),
                iov_len: 0,
            }; MAX_IOVECS_PER_WRITE],
            inflight_iov_count: 0,
        }
    }

    fn read_buf_tail(&mut self) -> (*mut u8, u32) {
        (
            unsafe { self.read_buf.as_mut_ptr().add(self.read_len) },
            (READ_BUF_SIZE - self.read_len) as u32,
        )
    }

    fn should_reap(&self, registry: &ConnectionRegistry) -> bool {
        self.read_closed
            && self.write_closed
            && !self.write_inflight
            && self.queue.is_empty()
            && self.inflight.is_empty()
            && registry.is_retired(self.conn)
    }
}

pub struct IngressThread<P> {
    thread_id: u8,
    listen_fd: RawFd,
    producer: P,
    allocator: PoolAllocator,
    response_queue: Arc<ResponseQueue>,
    publish_gate: Arc<Mutex<()>>,
    registry: Arc<ConnectionRegistry>,
}

impl<P> IngressThread<P>
where
    P: Producer<InferenceEvent>,
{
    pub fn new(
        thread_id: u8,
        listen_fd: RawFd,
        producer: P,
        allocator: PoolAllocator,
        response_queue: Arc<ResponseQueue>,
        publish_gate: Arc<Mutex<()>>,
        registry: Arc<ConnectionRegistry>,
    ) -> Self {
        Self {
            thread_id,
            listen_fd,
            producer,
            allocator,
            response_queue,
            publish_gate,
            registry,
        }
    }

    pub fn run(mut self) {
        let mut ring = IoUring::new(4096).expect("io_uring creation failed");
        let mut conns: Slab<Connection> = Slab::with_capacity(SLAB_CAPACITY);
        let mut cqe_buf: Vec<(u64, i32)> = Vec::new();
        let mut parse_queue: VecDeque<u16> = VecDeque::new();
        let mut parse_submit_budget = 0u8;
        submit_accept(&mut ring, self.listen_fd);
        submit_notify(&mut ring, self.response_queue.notify_fd());

        loop {
            drain_response_queue(&mut conns, &self.response_queue);
            submit_ready_writes(&mut ring, &mut conns, &self.registry);

            if let Some(key) = parse_queue.pop_front() {
                if let Some(conn) = conns.get_mut(key as usize) {
                    conn.parse_queued = false;
                }
                parse_and_maybe_read(
                    &mut ring,
                    &mut conns,
                    &mut parse_queue,
                    &mut self.producer,
                    &mut self.allocator,
                    &self.publish_gate,
                    &self.registry,
                    key,
                );
                // Queued parse work can enqueue follow-on reads and shard-local writes.
                // Submit often enough to preserve progress, but batch a few parse iterations
                // together so the hot path does not pay an `io_uring_enter` syscall on every
                // single buffered parse step.
                parse_submit_budget = parse_submit_budget.saturating_add(1);
                if parse_queue.is_empty() || parse_submit_budget >= 8 {
                    ring.submit();
                    parse_submit_budget = 0;
                }
                reap_retired_connections(&mut conns, &self.registry);
                continue;
            }

            parse_submit_budget = 0;
            ring.wait(1);
            cqe_buf.clear();
            ring.drain_cqes_into(&mut cqe_buf);

            for &(user_data, result) in &cqe_buf {
                let (op, data) = decode_user_data(user_data);
                match op {
                    OP_ACCEPT => handle_accept(
                        &mut ring,
                        &mut conns,
                        result,
                        self.thread_id,
                        self.listen_fd,
                        &self.registry,
                    ),
                    OP_READ => handle_read(
                        &mut ring,
                        &mut conns,
                        &mut parse_queue,
                        &mut self.producer,
                        &mut self.allocator,
                        &self.publish_gate,
                        &self.registry,
                        data as u16,
                        result,
                    ),
                    OP_WRITE => handle_write(&mut conns, &self.registry, data as u16, result),
                    OP_NOTIFY => handle_notify(&mut ring, self.response_queue.notify_fd(), result),
                    _ => {}
                }
            }

            reap_retired_connections(&mut conns, &self.registry);
        }
    }
}

fn drain_response_queue(conns: &mut Slab<Connection>, response_queue: &Arc<ResponseQueue>) {
    while let Some(response) = response_queue.pop() {
        let Some(conn) = conns.get_mut(response.conn.conn_id as usize) else {
            continue;
        };
        if conn.conn != response.conn || conn.write_closed {
            continue;
        }
        conn.queue.push_back(Box::new(ResponseFrame::new(
            response.published_at_ns,
            &response.data[..response.len],
        )));
        conn.ready_queued = true;
    }
}

fn reap_retired_connections(conns: &mut Slab<Connection>, registry: &Arc<ConnectionRegistry>) {
    let retired: Vec<u16> = conns
        .iter()
        .filter_map(|(k, c)| c.should_reap(registry).then_some(k as u16))
        .collect();
    for key in retired {
        conns.try_remove(key as usize);
    }
}

fn maybe_mark_read_closed(registry: &Arc<ConnectionRegistry>, conn: &mut Connection) {
    if conn.read_closed
        && !conn.write_inflight
        && conn.queue.is_empty()
        && conn.inflight.is_empty()
        && !conn.write_closed
    {
        conn.write_closed = true;
        registry.mark_read_closed(conn.conn, conn.next_request_seq);
    }
}

fn handle_accept(
    ring: &mut IoUring,
    conns: &mut Slab<Connection>,
    result: i32,
    thread_id: u8,
    listen_fd: RawFd,
    registry: &Arc<ConnectionRegistry>,
) {
    if result >= 0 {
        let client_fd = result as RawFd;
        if conns.len() >= SLAB_CAPACITY {
            unsafe { libc::close(client_fd) };
        } else {
            let entry = conns.vacant_entry();
            let key = entry.key();
            let conn = registry.open(thread_id, key as u16, client_fd);
            entry.insert(Connection::new(client_fd, conn));
            submit_read(ring, conns, key as u16);
        }
    }
    submit_accept(ring, listen_fd);
}

#[allow(clippy::too_many_arguments)]
fn handle_read(
    ring: &mut IoUring,
    conns: &mut Slab<Connection>,
    parse_queue: &mut VecDeque<u16>,
    producer: &mut impl Producer<InferenceEvent>,
    allocator: &mut PoolAllocator,
    publish_gate: &Arc<Mutex<()>>,
    registry: &Arc<ConnectionRegistry>,
    key: u16,
    result: i32,
) {
    let key_usize = key as usize;
    metrics::inc_read_cqes();
    if result <= 0 {
        if result < 0 {
            metrics::inc_read_negative();
        }
        if let Some(conn) = conns.get_mut(key_usize) {
            conn.read_inflight = false;
            conn.read_closed = true;
            maybe_mark_read_closed(registry, conn);
        }
        return;
    }

    let bytes_read = result as usize;
    metrics::add_read_bytes(bytes_read as u64);
    let Some(conn) = conns.get_mut(key_usize) else {
        return;
    };
    conn.read_inflight = false;
    conn.read_len += bytes_read;
    enqueue_parse(conns, parse_queue, key);

    parse_and_maybe_read(
        ring,
        conns,
        parse_queue,
        producer,
        allocator,
        publish_gate,
        registry,
        key,
    );
}

#[allow(clippy::too_many_arguments)]
fn parse_and_maybe_read(
    ring: &mut IoUring,
    conns: &mut Slab<Connection>,
    parse_queue: &mut VecDeque<u16>,
    producer: &mut impl Producer<InferenceEvent>,
    allocator: &mut PoolAllocator,
    publish_gate: &Arc<Mutex<()>>,
    registry: &Arc<ConnectionRegistry>,
    key: u16,
) {
    let key_usize = key as usize;
    let Some(conn) = conns.get_mut(key_usize) else {
        return;
    };
    let buf = &conn.read_buf[..conn.read_len];

    let publish_guard = publish_gate.lock().unwrap();
    match request_flow::process_requests_from_buffer(
        buf,
        producer,
        allocator,
        conn.conn,
        &mut conn.next_request_seq,
    ) {
        Ok(outcome) => {
            drop(publish_guard);
            if outcome.consumed > 0 {
                conn.read_buf
                    .copy_within(outcome.consumed..conn.read_len, 0);
                conn.read_len -= outcome.consumed;
            }
            metrics::add_bytes_consumed(outcome.consumed as u64);
            registry.update_published_seq_end(conn.conn, conn.next_request_seq);
            if outcome.needs_read {
                submit_read(ring, conns, key);
            }
        }
        Err(e) => {
            drop(publish_guard);
            eprintln!(
                "io-{}: request parse error ({:?}), closing conn {}",
                conn.conn.shard_id(),
                e,
                key
            );
            conn.read_closed = true;
            maybe_mark_read_closed(registry, conn);
            return;
        }
    }

    let c = &conns[key as usize];
    if !c.read_closed && c.read_len == 0 {
        submit_read(ring, conns, key);
    } else if !c.read_closed && !c.read_inflight && c.read_len > 0 {
        enqueue_parse(conns, parse_queue, key);
    }
}

fn enqueue_parse(conns: &mut Slab<Connection>, parse_queue: &mut VecDeque<u16>, key: u16) {
    let Some(conn) = conns.get_mut(key as usize) else {
        return;
    };
    if conn.read_closed || conn.read_inflight || conn.read_len == 0 || conn.parse_queued {
        return;
    }
    conn.parse_queued = true;
    parse_queue.push_back(key);
}

fn submit_accept(ring: &mut IoUring, listen_fd: RawFd) {
    let sqe = opcode::Accept::new(Fd(listen_fd), ptr::null_mut(), ptr::null_mut())
        .build()
        .user_data(encode_user_data(OP_ACCEPT, 0));
    ring.push(&sqe);
}

fn submit_notify(ring: &mut IoUring, notify_fd: RawFd) {
    let sqe = opcode::PollAdd::new(Fd(notify_fd), libc::POLLIN as _)
        .build()
        .user_data(encode_user_data(OP_NOTIFY, 0));
    ring.push(&sqe);
}

fn handle_notify(ring: &mut IoUring, notify_fd: RawFd, result: i32) {
    if result >= 0 {
        loop {
            let mut value = 0u64;
            let rc = unsafe {
                libc::read(
                    notify_fd,
                    (&mut value as *mut u64).cast::<libc::c_void>(),
                    std::mem::size_of::<u64>(),
                )
            };
            if rc < 0 {
                let err = std::io::Error::last_os_error()
                    .raw_os_error()
                    .unwrap_or_default();
                if err == libc::EAGAIN {
                    break;
                }
                panic!("eventfd read failed: {err}");
            }
            if rc == 0 {
                break;
            }
        }
    }
    submit_notify(ring, notify_fd);
}

fn submit_read(ring: &mut IoUring, conns: &mut Slab<Connection>, key: u16) {
    let conn = &mut conns[key as usize];
    if conn.read_inflight || conn.read_closed {
        return;
    }
    conn.read_inflight = true;
    let (buf_ptr, buf_len) = conn.read_buf_tail();
    let sqe = opcode::Recv::new(Fd(conn.fd), buf_ptr, buf_len)
        .build()
        .user_data(encode_user_data(OP_READ, key as u32));
    ring.push(&sqe);
    metrics::inc_read_submits();
}

fn submit_ready_writes(
    ring: &mut IoUring,
    conns: &mut Slab<Connection>,
    registry: &Arc<ConnectionRegistry>,
) {
    let ready: Vec<u16> = conns
        .iter()
        .filter_map(|(key, conn)| {
            (conn.ready_queued && !conn.write_closed && !conn.write_inflight).then_some(key as u16)
        })
        .collect();
    for key in ready {
        submit_write(ring, conns, registry, key);
    }
}

fn submit_write(
    ring: &mut IoUring,
    conns: &mut Slab<Connection>,
    registry: &Arc<ConnectionRegistry>,
    key: u16,
) {
    let conn = &mut conns[key as usize];
    conn.ready_queued = false;
    if conn.write_closed || conn.write_inflight {
        return;
    }

    if conn.inflight.is_empty() {
        while conn.inflight.len() < MAX_IOVECS_PER_WRITE {
            let Some(frame) = conn.queue.pop_front() else {
                break;
            };
            metrics::record_publish_to_write_submit(elapsed_since_ns(frame.published_at_ns));
            conn.inflight.push_back(frame);
        }
    }

    if conn.inflight.is_empty() {
        maybe_mark_read_closed(registry, conn);
        return;
    }

    let mut iov_count = 0usize;
    for frame in conn.inflight.iter() {
        conn.inflight_iovecs[iov_count] = libc::iovec {
            iov_base: frame.remaining_ptr() as *mut libc::c_void,
            iov_len: frame.remaining(),
        };
        iov_count += 1;
    }
    conn.inflight_iov_count = iov_count;
    conn.write_inflight = true;

    let sqe = opcode::Writev::new(Fd(conn.fd), conn.inflight_iovecs.as_ptr(), iov_count as u32)
        .build()
        .user_data(encode_user_data(OP_WRITE, key as u32));
    ring.push(&sqe);
    metrics::inc_write_sqes();
}

fn handle_write(
    conns: &mut Slab<Connection>,
    registry: &Arc<ConnectionRegistry>,
    key: u16,
    result: i32,
) {
    metrics::inc_write_cqes();
    let Some(conn) = conns.get_mut(key as usize) else {
        return;
    };
    if result < 0 {
        metrics::inc_write_negative();
        match -result {
            libc::EPIPE | libc::EBADF | libc::ECONNRESET => {
                metrics::inc_write_fatal();
            }
            libc::EAGAIN => {
                metrics::inc_write_eagain();
                metrics::inc_write_fatal();
            }
            _ => {}
        }
        conn.write_closed = true;
        conn.write_inflight = false;
        conn.inflight.clear();
        conn.queue.clear();
        conn.inflight_iov_count = 0;
        conn.read_closed = true;
        maybe_mark_read_closed(registry, conn);
        return;
    }

    let mut remaining = result as usize;
    while remaining > 0 {
        let Some(frame) = conn.inflight.front_mut() else {
            break;
        };
        let frame_remaining = frame.remaining();
        if remaining >= frame_remaining {
            remaining -= frame_remaining;
            conn.inflight.pop_front();
        } else {
            frame.offset += remaining;
            remaining = 0;
        }
    }

    conn.write_inflight = false;
    conn.inflight_iov_count = 0;

    if !conn.inflight.is_empty() || !conn.queue.is_empty() {
        conn.ready_queued = true;
    } else {
        maybe_mark_read_closed(registry, conn);
    }
}
