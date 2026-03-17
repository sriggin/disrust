use std::os::fd::RawFd;
use std::sync::Mutex;

use crate::config::SLAB_CAPACITY;
use crate::connection_id::{ConnectionRef, MAX_GENERATION};

struct ConnectionWriteState {
    generation: u16,
    fd: RawFd,
    read_closed: bool,
    retired: bool,
    published_seq_end: u64,
}

impl ConnectionWriteState {
    fn new() -> Self {
        Self {
            generation: 0,
            fd: -1,
            read_closed: false,
            retired: true,
            published_seq_end: 0,
        }
    }

    fn open(&mut self, fd: RawFd) -> u16 {
        self.generation = if self.generation >= MAX_GENERATION {
            1
        } else {
            self.generation + 1
        };
        self.fd = fd;
        self.read_closed = false;
        self.retired = false;
        self.published_seq_end = 0;
        self.generation
    }

    fn matches(&self, conn: ConnectionRef) -> bool {
        !self.retired && self.generation == conn.generation()
    }

    fn maybe_retire(&mut self) {
        if self.retired {
            return;
        }
        if self.read_closed {
            if self.fd >= 0 {
                unsafe {
                    libc::close(self.fd);
                }
            }
            self.fd = -1;
            self.retired = true;
        }
    }
}

struct Slot {
    state: Mutex<ConnectionWriteState>,
}

pub struct ConnectionRegistry {
    slots: Box<[Slot]>,
    shard_capacity: usize,
}

impl ConnectionRegistry {
    pub fn new(shard_count: usize, shard_capacity: usize) -> Self {
        let slots = (0..(shard_count * shard_capacity))
            .map(|_| Slot {
                state: Mutex::new(ConnectionWriteState::new()),
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self {
            slots,
            shard_capacity,
        }
    }

    fn slot_index(&self, conn: ConnectionRef) -> usize {
        conn.shard_id() as usize * self.shard_capacity + conn.conn_id as usize
    }

    fn slot_index_parts(&self, shard_id: u8, conn_id: u16) -> usize {
        shard_id as usize * self.shard_capacity + conn_id as usize
    }

    pub fn open(&self, shard_id: u8, conn_id: u16, fd: RawFd) -> ConnectionRef {
        let mut state = self.slots[self.slot_index_parts(shard_id, conn_id)]
            .state
            .lock()
            .unwrap();
        assert!(
            state.retired,
            "connection slot {}:{} reopened before retirement",
            shard_id, conn_id
        );
        let generation = state.open(fd);
        ConnectionRef::new(shard_id, conn_id, generation)
    }

    pub fn update_published_seq_end(&self, conn: ConnectionRef, published_seq_end: u64) {
        let mut state = self.slots[self.slot_index(conn)].state.lock().unwrap();
        if state.matches(conn) {
            state.published_seq_end = published_seq_end.max(state.published_seq_end);
        }
    }

    pub fn mark_read_closed(&self, conn: ConnectionRef, published_seq_end: u64) {
        let mut state = self.slots[self.slot_index(conn)].state.lock().unwrap();
        if !state.matches(conn) {
            return;
        }
        state.read_closed = true;
        state.published_seq_end = published_seq_end.max(state.published_seq_end);
        state.maybe_retire();
    }

    pub fn is_retired(&self, conn: ConnectionRef) -> bool {
        let state = self.slots[self.slot_index(conn)].state.lock().unwrap();
        state.generation == conn.generation() && state.retired
    }

    pub fn is_open(&self, conn: ConnectionRef) -> bool {
        let state = self.slots[self.slot_index(conn)].state.lock().unwrap();
        state.matches(conn) && !state.retired
    }
}

pub fn encode_user_data(conn: ConnectionRef) -> u64 {
    conn.as_u32() as u64
}

pub fn decode_user_data(user_data: u64) -> ConnectionRef {
    ConnectionRef::from_u32(user_data as u32)
}

impl Default for ConnectionRegistry {
    fn default() -> Self {
        Self::new(1, SLAB_CAPACITY)
    }
}

#[cfg(test)]
mod tests {
    use super::{ConnectionRegistry, decode_user_data, encode_user_data};
    use crate::connection_id::ConnectionRef;

    #[test]
    fn user_data_round_trip() {
        let conn = ConnectionRef::new(3, 17, 0x0bee);
        assert_eq!(decode_user_data(encode_user_data(conn)), conn);
    }

    #[test]
    fn retires_connection_after_read_close() {
        let registry = ConnectionRegistry::new(1, 4);
        let conn = registry.open(0, 1, 42);
        registry.mark_read_closed(conn, 1);
        assert!(registry.is_retired(conn));
    }

    #[test]
    fn stale_generation_is_rejected() {
        let registry = ConnectionRegistry::new(1, 2);
        let conn = registry.open(0, 0, 10);
        registry.mark_read_closed(conn, 0);
        let next_conn = registry.open(0, 0, 11);
        assert_ne!(conn.generation(), next_conn.generation());
        assert!(!registry.is_open(conn));
    }

    #[test]
    fn shard_identity_keeps_same_conn_id_slots_independent() {
        let registry = ConnectionRegistry::new(2, 2);
        let conn0 = registry.open(0, 1, 10);
        let conn1 = registry.open(1, 1, 11);

        assert!(registry.is_open(conn0));
        assert!(registry.is_open(conn1));
    }
}
