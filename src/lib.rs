//! Library crate for disrust: batch processor, buffer pool, protocol, request/response flow, etc.
//!
//! The **binary** (`main.rs`) is the only io_uring entrypoint: it compiles `io_thread` and spawns
//! the IO thread and batch processor. `io_thread` is intentionally not re-exported from the lib,
//! so the library remains testable without io_uring (e.g. request_flow and response_flow integration tests).

pub mod batch_processor;
pub mod buffer_pool;
pub mod constants;
pub mod metrics;
pub mod protocol;
pub mod request_flow;
pub mod response_flow;
pub mod response_queue;
pub mod ring_types;
