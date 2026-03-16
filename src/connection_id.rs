use crate::config::{MAX_IO_THREADS, SLAB_CAPACITY};

pub const SHARD_BITS: u16 = 4;
pub const GENERATION_BITS: u16 = 12;
pub const MAX_GENERATION: u16 = (1 << GENERATION_BITS) - 1;
const SHARD_SHIFT: u16 = GENERATION_BITS;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct ConnectionRef {
    pub conn_id: u16,
    pub shard_gen: u16,
}

impl ConnectionRef {
    pub fn new(shard_id: u8, conn_id: u16, generation: u16) -> Self {
        assert!(
            shard_id < (1 << SHARD_BITS) as u8,
            "shard_id must fit in {} bits",
            SHARD_BITS
        );
        assert!(
            generation > 0 && generation <= MAX_GENERATION,
            "generation must be in 1..={MAX_GENERATION}"
        );
        Self {
            conn_id,
            shard_gen: ((shard_id as u16) << SHARD_SHIFT) | generation,
        }
    }

    pub fn shard_id(self) -> u8 {
        (self.shard_gen >> SHARD_SHIFT) as u8
    }

    pub fn generation(self) -> u16 {
        self.shard_gen & MAX_GENERATION
    }

    pub fn as_u32(self) -> u32 {
        ((self.shard_gen as u32) << 16) | self.conn_id as u32
    }

    pub fn from_u32(value: u32) -> Self {
        Self {
            conn_id: value as u16,
            shard_gen: (value >> 16) as u16,
        }
    }
}

const _: () = assert!(MAX_IO_THREADS <= (1 << SHARD_BITS) as usize);
const _: () = assert!(SLAB_CAPACITY <= u16::MAX as usize);

#[cfg(test)]
mod tests {
    use super::ConnectionRef;

    #[test]
    fn round_trips_packed_identity() {
        let conn = ConnectionRef::new(7, 0x1234, 0x456);
        assert_eq!(conn.shard_id(), 7);
        assert_eq!(conn.conn_id, 0x1234);
        assert_eq!(conn.generation(), 0x456);
        assert_eq!(ConnectionRef::from_u32(conn.as_u32()), conn);
    }
}
