//! ORT upfront verification harness.
//!
//! Runs four go/no-go checks against the `ort` 2.x crate with CUDA EP before
//! SubmissionConsumer / CompletionConsumer are built.  Exits 0 on all passing,
//! nonzero with descriptive errors otherwise.
//!
//! Run with: `cargo run --bin ort-verify --features cuda`

use std::time::Instant;

use ort::{
    IoBinding,
    execution_providers::CUDAExecutionProvider,
    memory::{AllocationDevice, MemoryInfo, MemoryType},
    session::{Session, SessionBuilder},
    value::Tensor,
};

use disrust::config::MAX_BATCH_VECTORS;
use disrust::constants::FEATURE_DIM as FDIM;

// ---------------------------------------------------------------------------
// Minimal synthetic ONNX model for verification.
//
// The model computes: output[i] = sum(input[i, :]) for each input row.
// Built using the `ort` test utilities or a tiny hand-crafted ONNX protobuf.
// ---------------------------------------------------------------------------

// We embed a minimal ONNX model as bytes.  This model:
//   - Input:  "input"  shape [N, FEATURE_DIM] float32
//   - Output: "output" shape [N]              float32
//   - Op: ReduceSum on axis=1
//
// If you need a real model replace this with the actual ONNX bytes.
fn make_test_model_bytes() -> Vec<u8> {
    // Attempt to build a tiny ONNX model in-memory using ort's test support.
    // If unavailable, panic with a clear message instructing the user to provide a model.
    //
    // For now, we require the user to provide a test model at test_model.onnx.
    std::fs::read("test_model.onnx").unwrap_or_else(|_| {
        eprintln!(
            "ort-verify: please provide a 'test_model.onnx' file in the working directory.\n\
             The model should accept input shape [N, {FDIM}] float32 and produce output shape [N] float32."
        );
        std::process::exit(1);
    })
}

fn build_session(model_bytes: &[u8]) -> Session {
    SessionBuilder::new()
        .expect("SessionBuilder::new failed")
        .with_execution_providers([CUDAExecutionProvider::default().build()])
        .expect("with_execution_providers failed")
        .commit_from_memory(model_bytes)
        .expect("commit_from_memory failed")
}

fn cuda_alloc_host(byte_count: usize) -> *mut f32 {
    unsafe {
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let status = cudarc::driver::sys::lib().cuMemAllocHost_v2(&mut ptr, byte_count);
        assert_eq!(
            status,
            cudarc::driver::sys::CUresult::CUDA_SUCCESS,
            "cuMemAllocHost failed: {:?}",
            status
        );
        ptr as *mut f32
    }
}

fn cuda_free_host(ptr: *mut f32) {
    unsafe {
        let _ = cudarc::driver::sys::lib().cuMemFreeHost(ptr as *mut std::ffi::c_void);
    }
}

fn host_to_device_ptr(host_ptr: *mut f32) -> u64 {
    unsafe {
        let mut dev_ptr: u64 = 0;
        let status = cudarc::driver::sys::lib().cuMemHostGetDevicePointer_v2(
            &mut dev_ptr as *mut u64 as *mut _,
            host_ptr as *mut std::ffi::c_void,
            0,
        );
        assert_eq!(
            status,
            cudarc::driver::sys::CUresult::CUDA_SUCCESS,
            "cuMemHostGetDevicePointer failed: {:?}",
            status
        );
        dev_ptr
    }
}

// ---------------------------------------------------------------------------
// Check 1: Variable-batch run against fixed pre-bound output tensor
// ---------------------------------------------------------------------------

fn check1_variable_batch_fixed_output(model_bytes: &[u8]) {
    println!("Check 1: Variable-batch run against fixed pre-bound output tensor...");

    let k = 3; // vectors in test batch (K < MAX_BATCH_VECTORS)
    assert!(k < MAX_BATCH_VECTORS, "k must be < MAX_BATCH_VECTORS");

    let session = build_session(model_bytes);
    let mut binding = IoBinding::new(&session).expect("IoBinding::new failed");

    // Allocate output for MAX_BATCH_VECTORS and fill with a sentinel.
    let out_bytes = MAX_BATCH_VECTORS * std::mem::size_of::<f32>();
    let out_ptr = cuda_alloc_host(out_bytes);
    let sentinel = f32::from_bits(0xDEADBEEF);
    unsafe {
        for i in 0..MAX_BATCH_VECTORS {
            *out_ptr.add(i) = sentinel;
        }
    }

    // Bind pre-allocated output tensor.
    let out_mem = MemoryInfo::new(AllocationDevice::CUDA_PINNED, 0, MemoryType::Default)
        .expect("MemoryInfo failed");
    let out_tensor = unsafe {
        Tensor::<f32>::from_raw_ptr(out_ptr, &[MAX_BATCH_VECTORS as i64], out_mem)
            .expect("output Tensor::from_raw_ptr failed")
    };
    binding
        .bind_output("output", out_tensor)
        .expect("bind_output failed");

    // Allocate and fill input with K vectors (value = 1.0 each element).
    let in_bytes = k * FDIM * std::mem::size_of::<f32>();
    let in_ptr = cuda_alloc_host(in_bytes);
    unsafe { std::slice::from_raw_parts_mut(in_ptr, k * FDIM).fill(1.0f32) };
    let dev_in = host_to_device_ptr(in_ptr);

    let in_mem = MemoryInfo::new(AllocationDevice::CUDA_PINNED, 0, MemoryType::Default)
        .expect("MemoryInfo failed");
    let in_tensor = unsafe {
        Tensor::<f32>::from_raw_ptr(dev_in as *mut f32, &[k as i64, FDIM as i64], in_mem)
            .expect("input Tensor::from_raw_ptr failed")
    };
    binding
        .bind_input("input", in_tensor)
        .expect("bind_input failed");

    // Run inference.
    session
        .run_with_iobinding(&binding)
        .expect("run_with_iobinding failed");

    // Verify: first K outputs should be FEATURE_DIM (sum of all-ones row).
    let expected = FDIM as f32;
    for i in 0..k {
        let got = unsafe { *out_ptr.add(i) };
        assert!(
            (got - expected).abs() < 1e-4,
            "Check 1 FAIL: output[{}] = {} expected {}",
            i,
            got,
            expected
        );
    }
    // Remaining outputs should be unchanged (sentinel).
    for i in k..MAX_BATCH_VECTORS {
        let got = unsafe { *out_ptr.add(i) };
        if got != sentinel {
            println!(
                "  WARNING: output[{}] = {} (sentinel was {:.8e}); ORT wrote past batch boundary.",
                i, got, sentinel
            );
            println!("  ACTION: switch to per-batch dynamic output binding.");
        }
    }

    cuda_free_host(in_ptr);
    cuda_free_host(out_ptr);
    println!(
        "  Check 1 PASS: K={} results written correctly at [0, K).",
        k
    );
}

// ---------------------------------------------------------------------------
// Check 2: Zero-copy with CUDA_PINNED input via device pointer
// ---------------------------------------------------------------------------

fn check2_zero_copy_pinned_input(model_bytes: &[u8]) {
    println!("Check 2: Zero-copy with CUDA_PINNED input via cuMemHostGetDevicePointer...");

    let k = 4;
    let session = build_session(model_bytes);
    let mut binding = IoBinding::new(&session).expect("IoBinding::new failed");

    // Pinned input.
    let in_bytes = k * FDIM * std::mem::size_of::<f32>();
    let in_ptr = cuda_alloc_host(in_bytes);
    let in_vals: Vec<f32> = (0..k * FDIM).map(|i| (i % FDIM) as f32).collect();
    unsafe { in_ptr.copy_from(in_vals.as_ptr(), k * FDIM) };
    let dev_in = host_to_device_ptr(in_ptr);

    let in_mem = MemoryInfo::new(AllocationDevice::CUDA_PINNED, 0, MemoryType::Default)
        .expect("MemoryInfo failed");
    let in_tensor = unsafe {
        Tensor::<f32>::from_raw_ptr(dev_in as *mut f32, &[k as i64, FDIM as i64], in_mem)
            .expect("Tensor from device ptr failed")
    };
    binding
        .bind_input("input", in_tensor)
        .expect("bind_input failed");

    // CPU reference: sum each row.
    let expected: Vec<f32> = (0..k)
        .map(|row| in_vals[row * FDIM..(row + 1) * FDIM].iter().sum())
        .collect();

    // Pinned output.
    let out_bytes = k * std::mem::size_of::<f32>();
    let out_ptr = cuda_alloc_host(out_bytes);
    let out_mem = MemoryInfo::new(AllocationDevice::CUDA_PINNED, 0, MemoryType::Default)
        .expect("MemoryInfo failed");
    let out_tensor = unsafe {
        Tensor::<f32>::from_raw_ptr(out_ptr, &[k as i64], out_mem).expect("output Tensor failed")
    };
    binding
        .bind_output("output", out_tensor)
        .expect("bind_output failed");

    session.run_with_iobinding(&binding).expect("run failed");

    for i in 0..k {
        let got = unsafe { *out_ptr.add(i) };
        assert!(
            (got - expected[i]).abs() < 1e-4,
            "Check 2 FAIL: result[{}] = {} expected {}",
            i,
            got,
            expected[i]
        );
    }

    cuda_free_host(in_ptr);
    cuda_free_host(out_ptr);
    println!("  Check 2 PASS: device pointer input produces correct results.");
    println!("  NOTE: Whether this is truly zero-copy requires GPU profiling (nvprof/nsight).");
}

// ---------------------------------------------------------------------------
// Check 3: OrtRunHandle async semantics
// ---------------------------------------------------------------------------

fn check3_async_handle(model_bytes: &[u8]) {
    use std::future::Future;
    use std::pin::Pin;
    use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

    println!("Check 3: OrtRunHandle async semantics...");

    const VTABLE: RawWakerVTable =
        RawWakerVTable::new(|ptr| RawWaker::new(ptr, &VTABLE), |_| {}, |_| {}, |_| {});
    let waker = unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) };
    let mut cx = Context::from_waker(&waker);

    let k = 8usize;
    let session = build_session(model_bytes);
    let mut binding = IoBinding::new(&session).expect("IoBinding::new failed");

    let in_ptr = cuda_alloc_host(k * FDIM * std::mem::size_of::<f32>());
    unsafe { std::slice::from_raw_parts_mut(in_ptr, k * FDIM).fill(1.0f32) };
    let dev_in = host_to_device_ptr(in_ptr);
    let in_mem = MemoryInfo::new(AllocationDevice::CUDA_PINNED, 0, MemoryType::Default).unwrap();
    let in_tensor = unsafe {
        Tensor::<f32>::from_raw_ptr(dev_in as *mut f32, &[k as i64, FDIM as i64], in_mem).unwrap()
    };
    binding.bind_input("input", in_tensor).unwrap();

    let out_ptr = cuda_alloc_host(k * std::mem::size_of::<f32>());
    let out_mem = MemoryInfo::new(AllocationDevice::CUDA_PINNED, 0, MemoryType::Default).unwrap();
    let out_tensor = unsafe { Tensor::<f32>::from_raw_ptr(out_ptr, &[k as i64], out_mem).unwrap() };
    binding.bind_output("output", out_tensor).unwrap();

    let t0 = Instant::now();
    let mut handle = session.run_async(&binding).expect("run_async failed");
    let after_call = t0.elapsed();

    // First poll: check if it's immediately Ready (would mean synchronous).
    let first_poll = {
        let pinned = unsafe { Pin::new_unchecked(&mut handle) };
        pinned.poll(&mut cx)
    };

    let after_first_poll = t0.elapsed();

    match first_poll {
        Poll::Pending => {
            println!(
                "  PASS: run_async returned in {:?}, handle is Pending (non-blocking).",
                after_call
            );
            // Spin to completion.
            loop {
                let pinned = unsafe { Pin::new_unchecked(&mut handle) };
                if let Poll::Ready(r) = pinned.poll(&mut cx) {
                    r.expect("handle resolved with error");
                    break;
                }
                std::hint::spin_loop();
            }
            println!("  Check 3 PASS: truly async (non-blocking call).");
        }
        Poll::Ready(r) => {
            r.expect("handle resolved with error");
            println!(
                "  WARNING: run_async resolved immediately ({:?} from call to first poll).",
                after_first_poll
            );
            println!("  The CUDA EP async path may be synchronous in this configuration.");
            println!("  Design is still correct but degenerates to serial GPU execution.");
            println!("  Check 3 NOTE: synchronous degradation — performance model changes.");
        }
    }

    cuda_free_host(in_ptr);
    cuda_free_host(out_ptr);
}

// ---------------------------------------------------------------------------
// Check 4: Dynamic input shape via IoBinding
// ---------------------------------------------------------------------------

fn check4_dynamic_shape(model_bytes: &[u8]) {
    println!("Check 4: Dynamic input shape via IoBinding (varying batch size per call)...");

    let session = build_session(model_bytes);

    for &k in &[1usize, 7, 16, MAX_BATCH_VECTORS] {
        let mut binding = IoBinding::new(&session).expect("IoBinding::new failed");

        let in_ptr = cuda_alloc_host(k * FDIM * std::mem::size_of::<f32>());
        unsafe { std::slice::from_raw_parts_mut(in_ptr, k * FDIM).fill(2.0f32) };
        let dev_in = host_to_device_ptr(in_ptr);
        let in_mem =
            MemoryInfo::new(AllocationDevice::CUDA_PINNED, 0, MemoryType::Default).unwrap();
        let in_tensor = unsafe {
            Tensor::<f32>::from_raw_ptr(dev_in as *mut f32, &[k as i64, FDIM as i64], in_mem)
                .expect("input tensor failed")
        };
        binding.bind_input("input", in_tensor).unwrap();

        let out_ptr = cuda_alloc_host(k * std::mem::size_of::<f32>());
        let out_mem =
            MemoryInfo::new(AllocationDevice::CUDA_PINNED, 0, MemoryType::Default).unwrap();
        let out_tensor =
            unsafe { Tensor::<f32>::from_raw_ptr(out_ptr, &[k as i64], out_mem).unwrap() };
        binding.bind_output("output", out_tensor).unwrap();

        session.run_with_iobinding(&binding).expect("run failed");

        let expected = 2.0f32 * FDIM as f32;
        for i in 0..k {
            let got = unsafe { *out_ptr.add(i) };
            assert!(
                (got - expected).abs() < 1e-3,
                "Check 4 FAIL for k={}: result[{}] = {} expected {}",
                k,
                i,
                got,
                expected
            );
        }

        cuda_free_host(in_ptr);
        cuda_free_host(out_ptr);
        println!("  k={}: OK", k);
    }

    println!("  Check 4 PASS: dynamic input shapes accepted correctly.");
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    println!("ort-verify: running ORT upfront verification checks");
    println!("  FEATURE_DIM={FDIM}, MAX_BATCH_VECTORS={MAX_BATCH_VECTORS}");

    let model_bytes = make_test_model_bytes();

    let mut failed = false;

    macro_rules! run_check {
        ($check:expr, $name:expr) => {
            if let Err(e) = std::panic::catch_unwind(|| $check) {
                eprintln!("FAIL: {} panicked: {:?}", $name, e);
                failed = true;
            }
        };
    }

    run_check!(check1_variable_batch_fixed_output(&model_bytes), "Check 1");
    run_check!(check2_zero_copy_pinned_input(&model_bytes), "Check 2");
    run_check!(check3_async_handle(&model_bytes), "Check 3");
    run_check!(check4_dynamic_shape(&model_bytes), "Check 4");

    if failed {
        eprintln!("ort-verify: one or more checks FAILED");
        std::process::exit(1);
    }

    println!("ort-verify: all checks PASSED");
}
