//! Library crate for disrust: ONNX inference server support, request parsing, buffer pool, and shared types.
//!
//! The `disrust` binary is the only io_uring server entrypoint. The library intentionally exposes
//! the protocol, request path, and pipeline pieces so they can be tested without starting the
//! full network server.

pub mod affinity;
pub mod buffer_pool;
pub mod clock;
pub mod config;
pub mod connection_id;
pub mod constants;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod metrics;
pub mod pipeline;
pub mod protocol;
pub mod request_flow;
pub mod ring_types;
pub mod server;
pub mod timer;
pub mod verify;
