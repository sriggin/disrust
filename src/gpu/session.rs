//! ORT session wrapper with pinned host memory I/O tensors.

use ort::{
    IoBinding,
    execution_providers::CUDAExecutionProvider,
    memory::{AllocationDevice, MemoryInfo, MemoryType},
    session::{Session, SessionBuilder, run::OrtRunHandle},
    value::Tensor,
};

use crate::config::MAX_BATCH_VECTORS;
use crate::constants::FEATURE_DIM as FDIM;

/// Per-session state: one ORT session, one pre-bound output tensor.
pub struct GpuSession {
    session: Session,
    io_binding: ort::IoBinding,
    /// Pinned host output buffer: `[MAX_BATCH_VECTORS]` f32.
    /// Bound once at construction; ORT overwrites in-place each batch run.
    output_host_ptr: *mut f32,
    /// Length in f32 elements (== MAX_BATCH_VECTORS).
    output_len: usize,
}

// SAFETY: GpuSession is moved to its owning thread and never shared concurrently.
unsafe impl Send for GpuSession {}

impl GpuSession {
    /// Construct a session for the given ONNX model bytes.
    ///
    /// Allocates a pinned output tensor via `cudaMallocHost` and pre-binds it
    /// so each batch run overwrites the tensor in-place.
    ///
    /// # Panics / Aborts
    /// Any ORT or CUDA error calls `std::process::abort()`.  GPU errors are
    /// unrecoverable — the process is restarted by the supervisor.
    pub fn new(model_bytes: &[u8]) -> Self {
        let output_len = MAX_BATCH_VECTORS;
        let byte_count = output_len * std::mem::size_of::<f32>();

        // Allocate pinned host memory for the output tensor.
        let output_host_ptr: *mut f32 = unsafe {
            let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            let status = cudarc::driver::sys::lib().cuMemAllocHost_v2(&mut ptr, byte_count);
            if status != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                eprintln!("cuMemAllocHost failed: {:?}", status);
                std::process::abort();
            }
            ptr as *mut f32
        };

        // Zero-initialise the output buffer.
        unsafe { std::ptr::write_bytes(output_host_ptr, 0, output_len) };

        // Build ORT session with CUDA EP.
        let session = SessionBuilder::new()
            .unwrap_or_else(|e| {
                eprintln!("SessionBuilder::new failed: {e}");
                std::process::abort()
            })
            .with_execution_providers([CUDAExecutionProvider::default().build()])
            .unwrap_or_else(|e| {
                eprintln!("with_execution_providers failed: {e}");
                std::process::abort()
            })
            .commit_from_memory(model_bytes)
            .unwrap_or_else(|e| {
                eprintln!("commit_from_memory failed: {e}");
                std::process::abort()
            });

        // Build IoBinding and pre-bind the pinned output tensor.
        let mut io_binding = IoBinding::new(&session).unwrap_or_else(|e| {
            eprintln!("IoBinding::new failed: {e}");
            std::process::abort()
        });

        let output_memory_info =
            MemoryInfo::new(AllocationDevice::CUDA_PINNED, 0, MemoryType::Default).unwrap_or_else(
                |e| {
                    eprintln!("MemoryInfo for output failed: {e}");
                    std::process::abort()
                },
            );

        // Pre-bind output tensor with shape [MAX_BATCH_VECTORS].
        let output_tensor = unsafe {
            Tensor::<f32>::from_raw_ptr(output_host_ptr, &[output_len as i64], output_memory_info)
                .unwrap_or_else(|e| {
                    eprintln!("output Tensor::from_raw_ptr failed: {e}");
                    std::process::abort()
                })
        };

        io_binding
            .bind_output("output", output_tensor)
            .unwrap_or_else(|e| {
                eprintln!("bind_output failed: {e}");
                std::process::abort()
            });

        Self {
            session,
            io_binding,
            output_host_ptr,
            output_len,
        }
    }

    /// Submit a batch for async GPU inference.
    ///
    /// `device_input_ptr` is the CUDA device pointer for the batch's input data
    /// (contiguous `[num_vectors × FEATURE_DIM]` f32s in pinned host memory).
    /// `num_vectors` is the total vector count across all slots in this batch.
    ///
    /// Returns an `OrtRunHandle` that resolves after GPU completion and the
    /// output tensor is fully written.
    pub fn run_async(&mut self, device_input_ptr: u64, num_vectors: usize) -> OrtRunHandle {
        let input_memory_info =
            MemoryInfo::new(AllocationDevice::CUDA_PINNED, 0, MemoryType::Default).unwrap_or_else(
                |e| {
                    eprintln!("MemoryInfo for input failed: {e}");
                    std::process::abort()
                },
            );

        let input_tensor = unsafe {
            Tensor::<f32>::from_raw_ptr(
                device_input_ptr as *mut f32,
                &[num_vectors as i64, FDIM as i64],
                input_memory_info,
            )
            .unwrap_or_else(|e| {
                eprintln!("input Tensor::from_raw_ptr failed: {e}");
                std::process::abort()
            })
        };

        self.io_binding
            .bind_input("input", input_tensor)
            .unwrap_or_else(|e| {
                eprintln!("bind_input failed: {e}");
                std::process::abort()
            });

        self.session
            .run_async(&self.io_binding)
            .unwrap_or_else(|e| {
                eprintln!("run_async failed: {e}");
                std::process::abort()
            })
    }

    /// Read the output tensor written by the last completed GPU batch.
    ///
    /// Returns the full pre-allocated buffer `[MAX_BATCH_VECTORS]` f32s.
    /// The caller is responsible for indexing only the valid prefix.
    pub fn output_slice(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.output_host_ptr, self.output_len) }
    }

    /// Return the raw output pointer and element count.
    ///
    /// Allows a separate consumer thread to read results without shared ownership
    /// of the GpuSession — the pointer is valid for the session's lifetime.
    ///
    /// # Safety
    /// The caller must only read from this pointer after awaiting the corresponding
    /// `OrtRunHandle`, which guarantees GPU writes are complete.
    pub unsafe fn output_raw_ptr(&self) -> (*const f32, usize) {
        (self.output_host_ptr, self.output_len)
    }
}

impl Drop for GpuSession {
    fn drop(&mut self) {
        // Free the pinned output buffer.  Failure is logged but not fatal on drop.
        unsafe {
            let status = cudarc::driver::sys::lib()
                .cuMemFreeHost(self.output_host_ptr as *mut std::ffi::c_void);
            if status != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                eprintln!("cuMemFreeHost failed on GpuSession drop: {:?}", status);
            }
        }
    }
}
