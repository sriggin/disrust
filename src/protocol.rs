use crate::ring_types::{FEATURE_DIM, MAX_VECTORS_PER_REQUEST};

/// Result of attempting to parse a request from a byte buffer.
#[allow(dead_code)]
pub enum ParseResult {
    /// Successfully parsed a request. Contains (num_vectors, total bytes consumed).
    Complete {
        num_vectors: u32,
        bytes_consumed: usize,
    },
    /// Need more data. Contains minimum bytes still needed.
    Incomplete(usize),
    /// Protocol error (e.g., num_vectors > MAX or == 0).
    Error(&'static str),
}

/// Try to parse a request from the buffer. Returns how many bytes were consumed
/// and the number of vectors. Feature data starts at offset 4 in the buffer.
pub fn try_parse_request(buf: &[u8]) -> ParseResult {
    if buf.len() < 4 {
        return ParseResult::Incomplete(4 - buf.len());
    }

    let num_vectors = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);

    if num_vectors == 0 || num_vectors as usize > MAX_VECTORS_PER_REQUEST {
        return ParseResult::Error("num_vectors out of range");
    }

    let payload_size = num_vectors as usize * FEATURE_DIM * 4;
    let total_size = 4 + payload_size;

    if buf.len() < total_size {
        return ParseResult::Incomplete(total_size - buf.len());
    }

    ParseResult::Complete {
        num_vectors,
        bytes_consumed: total_size,
    }
}

/// Copy feature data from a raw byte buffer (starting after the 4-byte header)
/// into the pre-allocated f32 slice in the disruptor event.
///
/// # Safety
/// Caller must ensure `src` has at least `num_vectors * FEATURE_DIM * 4` bytes
/// and `dst` has at least `num_vectors * FEATURE_DIM` f32 slots.
pub fn copy_features(src: &[u8], dst: &mut [f32], num_vectors: u32) {
    let count = num_vectors as usize * FEATURE_DIM;
    // SAFETY: f32 and [u8; 4] have the same size, and we're just reinterpreting
    // little-endian bytes as f32. This is safe on little-endian platforms.
    // For portability, use from_le_bytes per element.
    for (i, dst_elem) in dst.iter_mut().enumerate().take(count) {
        let offset = i * 4;
        *dst_elem = f32::from_le_bytes([
            src[offset],
            src[offset + 1],
            src[offset + 2],
            src[offset + 3],
        ]);
    }
}

/// Serialize a response into the write buffer.
/// Format: [u32 num_vectors] [f32 * num_vectors]
pub fn write_response(buf: &mut Vec<u8>, num_vectors: u32, results: &[f32]) {
    buf.extend_from_slice(&num_vectors.to_le_bytes());
    for &val in &results[..num_vectors as usize] {
        buf.extend_from_slice(&val.to_le_bytes());
    }
}
