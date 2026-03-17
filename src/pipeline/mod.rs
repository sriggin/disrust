//! Inference pipeline: always-compiled session management, submission, and completion.

pub mod connection_registry;
pub mod inference;
pub mod ready_queue;
pub mod session;
pub mod writer;

use std::path::PathBuf;

use crate::buffer_pool::BufferPool;
#[cfg(feature = "cuda")]
use crate::config::GPU_BUFFER_POOL_BYTES;
use crate::config::GPU_BUFFER_POOL_CAPACITY;

/// Allocate and leak a `BufferPool` of the standard server capacity.
///
/// CUDA: backed by pinned host memory (`cuMemAllocHost`), enabling zero-copy DMA.
/// CPU: backed by heap memory.
///
/// The returned reference is `'static` because the pool lives for the process lifetime.
pub fn make_pool() -> &'static BufferPool {
    #[cfg(feature = "cuda")]
    let ptr = unsafe {
        let mut raw: *mut std::ffi::c_void = std::ptr::null_mut();
        let status = cudarc::driver::sys::cuMemAllocHost_v2(&mut raw, GPU_BUFFER_POOL_BYTES);
        if status != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            eprintln!("make_pool: cuMemAllocHost_v2 failed: {status:?}");
            std::process::abort();
        }
        std::ptr::write_bytes(raw as *mut u8, 0u8, GPU_BUFFER_POOL_BYTES);
        raw as *mut f32
    };

    #[cfg(not(feature = "cuda"))]
    let ptr = Box::leak(vec![0f32; GPU_BUFFER_POOL_CAPACITY].into_boxed_slice()).as_mut_ptr();

    Box::leak(unsafe { BufferPool::from_raw_ptr(ptr, GPU_BUFFER_POOL_CAPACITY) })
}

/// Locate the ONNX Runtime shared library to pass to `ort::init_from`.
pub fn verify_ort_dylib_present() -> Result<PathBuf, String> {
    fn default_ort_paths() -> Vec<PathBuf> {
        let mut paths = Vec::new();
        if let Ok(cwd) = std::env::current_dir() {
            paths.push(cwd.join(".local/onnxruntime/current/lib/libonnxruntime.so"));
        }
        if let Ok(exe) = std::env::current_exe()
            && let Some(parent) = exe.parent()
        {
            paths.push(parent.join("libonnxruntime.so"));
        }
        paths.push(PathBuf::from("/lib/x86_64-linux-gnu/libonnxruntime.so"));
        paths.push(PathBuf::from("/usr/lib/x86_64-linux-gnu/libonnxruntime.so"));
        paths.push(PathBuf::from("/lib/libonnxruntime.so"));
        paths.push(PathBuf::from("/usr/lib/libonnxruntime.so"));
        paths
    }

    if let Ok(path) = std::env::var("ORT_DYLIB_PATH")
        && !path.is_empty()
    {
        let path = PathBuf::from(path);
        if path.exists() {
            return Ok(path);
        }
        return Err(format!(
            "ORT_DYLIB_PATH is set but does not exist: {}",
            path.display()
        ));
    }

    for path in default_ort_paths() {
        if path.exists() {
            return Ok(path);
        }
    }

    Err(format!(
        "libonnxruntime.so not found. Set ORT_DYLIB_PATH or place libonnxruntime.so next to the binary. Searched: {}",
        default_ort_paths()
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    ))
}
