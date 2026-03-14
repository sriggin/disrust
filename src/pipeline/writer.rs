use std::collections::VecDeque;
use std::io;
use std::sync::Arc;
use std::thread;

use io_uring::{opcode, squeue::Entry, types::Fd};

use crate::metrics;
use crate::pipeline::connection_registry::{
    ConnectionRegistry, FlushSubmission, WriteResult, decode_user_data, encode_user_data,
};
use crate::pipeline::ready_queue::{ConnectionRef, ReadyQueue};

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

    fn submit(&mut self) {
        if self.outstanding > 0 {
            self.inner.submit().expect("io_uring submit failed");
        }
    }

    fn wait(&mut self, min_complete: usize) {
        self.inner
            .submit_and_wait(min_complete)
            .expect("submit_and_wait failed");
    }

    fn drain_cqes(&mut self, mut on_cqe: impl FnMut(u64, i32)) {
        for cqe in self.inner.completion() {
            self.outstanding = self.outstanding.saturating_sub(1);
            on_cqe(cqe.user_data(), cqe.result());
        }
    }
}

pub struct WriterConsumer {
    ready_queue: Arc<ReadyQueue>,
    registry: Arc<ConnectionRegistry>,
    ring: IoUring,
    retry_queue: VecDeque<ConnectionRef>,
}

impl WriterConsumer {
    pub fn new(
        ready_queue: Arc<ReadyQueue>,
        registry: Arc<ConnectionRegistry>,
    ) -> io::Result<Self> {
        Ok(Self {
            ready_queue,
            registry,
            ring: IoUring::new(4096)?,
            retry_queue: VecDeque::new(),
        })
    }

    pub fn run(mut self) {
        let mut idle_loops = 0u32;
        loop {
            let mut progressed = false;
            while let Some(conn) = self.retry_queue.pop_front() {
                progressed = true;
                self.try_submit(conn);
            }
            while let Some(conn) = self.ready_queue.pop() {
                progressed = true;
                self.try_submit(conn);
            }

            {
                let registry = Arc::clone(&self.registry);
                let retry_queue = &mut self.retry_queue;
                self.ring.drain_cqes(|user_data, result| {
                    metrics::inc_write_cqes();
                    let conn = decode_user_data(user_data);
                    if result < 0 {
                        metrics::inc_write_negative();
                        match -result {
                            libc::EPIPE | libc::EBADF | libc::ECONNRESET => {
                                metrics::inc_write_fatal();
                            }
                            libc::EAGAIN => {
                                metrics::inc_write_eagain();
                            }
                            _ => {}
                        }
                    }
                    match registry.handle_write_result(conn, result) {
                        WriteResult::NeedsResubmit | WriteResult::ReadyAgain => {
                            retry_queue.push_back(conn);
                        }
                        WriteResult::Completed
                        | WriteResult::Error(_)
                        | WriteResult::Idle
                        | WriteResult::Stale => {}
                    }
                });
            }

            if self.ring.outstanding > 0 {
                idle_loops = 0;
                self.ring.wait(1);
                continue;
            }

            if !progressed {
                idle_loops = idle_loops.saturating_add(1);
                if idle_loops < 64 {
                    std::hint::spin_loop();
                } else {
                    thread::yield_now();
                }
            } else {
                idle_loops = 0;
            }
        }
    }

    fn try_submit(&mut self, conn: ConnectionRef) {
        let Some(FlushSubmission {
            fd,
            iovecs,
            iov_count,
        }) = self.registry.take_flush(conn)
        else {
            return;
        };

        let sqe = opcode::Writev::new(Fd(fd), iovecs, iov_count)
            .build()
            .user_data(encode_user_data(conn));
        self.ring.push(&sqe);
        self.ring.submit();
        metrics::inc_write_sqes();
    }
}
