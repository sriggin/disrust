//! Integration test: response path (guard → wire bytes per conn) without io_uring.

mod common;

use disrust::response_flow;
use disrust::ring_types::InferenceResponse;

#[test]
fn response_flow_guard_to_wire_per_conn_single_response() {
    let efd = common::create_eventfd();
    assert!(efd >= 0);

    let (mut producer, mut poller) = disrust::response_queue::build_response_channel(256, efd);

    let resp = InferenceResponse::with_results(1, 0, &[1.0f32, 2.0f32, 3.0f32], None).unwrap();
    producer.send(resp);

    match poller.poll() {
        Ok(mut guard) => {
            let wire = response_flow::guard_to_wire_per_conn(&mut guard);
            assert_eq!(wire.len(), 1);
            let buf = wire.get(&1).expect("conn_id 1");
            // [u8 num_vectors=3][f32; 3] = 1 + 12 = 13 bytes
            assert_eq!(buf.len(), 1 + 3 * 4);
            assert_eq!(buf[0], 3);
            assert_eq!(f32::from_le_bytes([buf[1], buf[2], buf[3], buf[4]]), 1.0);
            assert_eq!(f32::from_le_bytes([buf[5], buf[6], buf[7], buf[8]]), 2.0);
            assert_eq!(f32::from_le_bytes([buf[9], buf[10], buf[11], buf[12]]), 3.0);
        }
        Err(_) => panic!("expected one response"),
    }

    unsafe {
        libc::close(efd);
    }
}

#[test]
fn response_flow_guard_to_wire_per_conn_multiple_conns() {
    let efd = common::create_eventfd();
    assert!(efd >= 0);

    let (mut producer, mut poller) = disrust::response_queue::build_response_channel(256, efd);

    producer.send(InferenceResponse::with_results(1, 0, &[10.0f32], None).unwrap());
    producer.send(InferenceResponse::with_results(2, 0, &[20.0f32, 21.0f32], None).unwrap());
    producer.signal();

    match poller.poll() {
        Ok(mut guard) => {
            let wire = response_flow::guard_to_wire_per_conn(&mut guard);
            assert_eq!(wire.len(), 2);

            let buf1 = wire.get(&1).expect("conn_id 1");
            assert_eq!(buf1.len(), 1 + 4);
            assert_eq!(buf1[0], 1);
            assert_eq!(f32::from_le_bytes(buf1[1..5].try_into().unwrap()), 10.0);

            let buf2 = wire.get(&2).expect("conn_id 2");
            assert_eq!(buf2.len(), 1 + 8);
            assert_eq!(buf2[0], 2);
            assert_eq!(f32::from_le_bytes(buf2[1..5].try_into().unwrap()), 20.0);
            assert_eq!(f32::from_le_bytes(buf2[5..9].try_into().unwrap()), 21.0);
        }
        Err(_) => panic!("expected responses"),
    }

    unsafe {
        libc::close(efd);
    }
}
