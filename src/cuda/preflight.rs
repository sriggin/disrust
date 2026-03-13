use std::ffi::{CStr, c_char};

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
