//! GPU inference pipeline modules (CUDA EP via ONNX Runtime).
//! All items in this module require `--features cuda`.

pub mod batch_queue;
pub mod completion;
pub mod pinned;
pub mod preflight;
pub mod session;
pub mod submission;
