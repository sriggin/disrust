pub const FEATURE_DIM: usize = 16;
pub const MAX_VECTORS_PER_REQUEST: usize = 64;

const _: () = assert!(
    MAX_VECTORS_PER_REQUEST <= u8::MAX as usize,
    "num_vectors is u8"
);
