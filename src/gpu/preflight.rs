use std::ffi::{CStr, c_char};
use std::path::PathBuf;

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

pub fn verify_ort_dylib_present() -> Result<PathBuf, String> {
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

fn cuda_error_message(code: cudarc::driver::sys::CUresult) -> String {
    unsafe {
        let mut name: *const c_char = std::ptr::null();
        let mut message: *const c_char = std::ptr::null();
        let _ = cudarc::driver::sys::cuGetErrorName(code, &mut name);
        let _ = cudarc::driver::sys::cuGetErrorString(code, &mut message);
        let name = if name.is_null() {
            "UNKNOWN".to_string()
        } else {
            CStr::from_ptr(name).to_string_lossy().into_owned()
        };
        let message = if message.is_null() {
            "unknown CUDA error".to_string()
        } else {
            CStr::from_ptr(message).to_string_lossy().into_owned()
        };
        format!("{name} ({code:?}): {message}")
    }
}

pub fn verify_cuda_startup() -> Result<(), String> {
    unsafe {
        let init = cudarc::driver::sys::cuInit(0);
        if init != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(format!("cuInit failed: {}", cuda_error_message(init)));
        }

        let mut count = 0;
        let count_status = cudarc::driver::sys::cuDeviceGetCount(&mut count);
        if count_status != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(format!(
                "cuDeviceGetCount failed: {}",
                cuda_error_message(count_status)
            ));
        }
        if count <= 0 {
            return Err("cuDeviceGetCount succeeded but reported no CUDA devices".to_string());
        }
    }

    Ok(())
}
