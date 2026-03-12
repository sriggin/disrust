use std::ffi::c_void;

pub fn alloc_pinned(bytes: usize) -> Result<*mut c_void, String> {
    unsafe {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let status = cudarc::driver::sys::cuMemAllocHost_v2(&mut ptr, bytes);
        if status != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(format!("cuMemAllocHost_v2 failed: {status:?}"));
        }
        Ok(ptr)
    }
}

/// Free a pinned host allocation previously returned by [`alloc_pinned`].
///
/// # Safety
/// `ptr` must have been returned by `alloc_pinned`, must not already have been freed,
/// and must not be used after this call returns.
pub unsafe fn free_pinned(ptr: *mut c_void) -> Result<(), String> {
    let status = unsafe { cudarc::driver::sys::cuMemFreeHost(ptr) };
    if status != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
        return Err(format!("cuMemFreeHost failed: {status:?}"));
    }
    Ok(())
}
