//! Inference pipeline: always-compiled session management, submission, and completion.

pub mod batch_queue;
pub mod completion;
pub mod connection_registry;
pub mod inference;
pub mod ready_queue;
pub mod session;
pub mod submission;
pub mod writer;

pub use session::{InferenceBackend, OrtBackend};
