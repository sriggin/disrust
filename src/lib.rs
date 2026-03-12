//! Library crate for disrust: ONNX/CUDA server support, request parsing, buffer pool, and shared types.
//!
//! The `disrust` binary is the only io_uring server entrypoint. The library intentionally exposes
//! the protocol, request path, and GPU runtime pieces so they can be tested without starting the
//! full network server.

pub mod buffer_pool;
pub mod config;
pub mod constants;
pub mod metrics;
pub mod protocol;
pub mod request_flow;
pub mod ring_types;
#[cfg(feature = "cuda")]
pub mod batch_queue;
#[cfg(feature = "cuda")]
pub mod gpu;
