#![allow(dead_code)]

use std::os::unix::io::RawFd;

use disrust::buffer_pool::{BufferPool, set_factory_pool};
use disrust::constants::FEATURE_DIM;

pub fn init_factory_pool() {
    let _ = set_factory_pool(BufferPool::new_boxed(1));
}

pub fn create_eventfd() -> RawFd {
    unsafe { libc::eventfd(0, libc::EFD_NONBLOCK) }
}

/// Build a byte buffer for one request: [u32 num_vectors][f32 * num_vectors * FEATURE_DIM].
pub fn one_request_bytes(num_vectors: u32, feature_values: &[f32]) -> Vec<u8> {
    assert!(num_vectors as usize * FEATURE_DIM <= feature_values.len());
    let mut buf = num_vectors.to_le_bytes().to_vec();
    for val in feature_values
        .iter()
        .take(num_vectors as usize * FEATURE_DIM)
    {
        buf.extend_from_slice(&val.to_le_bytes());
    }
    buf
}
