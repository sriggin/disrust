//! ORT session wrapper backed by direct `ort-sys::RunAsync`.

use std::ffi::{CStr, CString, c_char, c_void};
use std::ptr::NonNull;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicU8, Ordering},
};

use ort::{
    AsPointer, api,
    memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType},
    session::Session,
    sys,
};

#[cfg(feature = "cuda")]
use ort::execution_providers::CUDAExecutionProvider;

use crate::buffer_pool::PoolSlice;
use crate::config::{MAX_BATCH_VECTORS, ORT_INTRA_THREADS};
use crate::constants::FEATURE_DIM as FDIM;

#[cfg(feature = "cuda")]
use crate::cuda::memory::{alloc_pinned, free_pinned};

const BATCH_PENDING: u8 = 0;
const BATCH_READY: u8 = 1;
const BATCH_FAILED: u8 = 2;

fn status_message(status: sys::OrtStatusPtr) -> String {
    if status.0.is_null() {
        return "unknown ORT error".to_string();
    }
    let api = api();
    let message = unsafe { (api.GetErrorMessage)(status.0) };
    let message = unsafe { CStr::from_ptr(message) }
        .to_string_lossy()
        .into_owned();
    unsafe { (api.ReleaseStatus)(status.0) };
    message
}

fn check_status(status: sys::OrtStatusPtr, context: &str) {
    if !status.0.is_null() {
        eprintln!("{context}: {}", status_message(status));
        std::process::abort();
    }
}

struct OrtValueHandle(NonNull<sys::OrtValue>);

impl OrtValueHandle {
    fn as_ptr(&self) -> *mut sys::OrtValue {
        self.0.as_ptr()
    }
}

impl Drop for OrtValueHandle {
    fn drop(&mut self) {
        unsafe { (api().ReleaseValue)(self.0.as_ptr()) };
    }
}

unsafe impl Send for OrtValueHandle {}

fn create_tensor_from_external(
    memory_info: &MemoryInfo,
    data: *mut c_void,
    elem_count: usize,
    shape: &[i64],
) -> OrtValueHandle {
    let mut value_ptr: *mut sys::OrtValue = std::ptr::null_mut();
    let status = unsafe {
        (api().CreateTensorWithDataAsOrtValue)(
            memory_info.ptr(),
            data,
            elem_count * std::mem::size_of::<f32>(),
            shape.as_ptr(),
            shape.len(),
            sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &mut value_ptr,
        )
    };
    check_status(status, "CreateTensorWithDataAsOrtValue failed");
    OrtValueHandle(NonNull::new(value_ptr).expect("ORT returned null OrtValue"))
}

/// Completion status for one submitted batch.
pub struct BatchCompletion {
    state: AtomicU8,
    error: Mutex<Option<String>>,
}

pub enum BatchPoll {
    Pending,
    Ready,
    Failed,
}

impl BatchCompletion {
    fn new() -> Self {
        Self {
            state: AtomicU8::new(BATCH_PENDING),
            error: Mutex::new(None),
        }
    }

    fn mark_ready(&self) {
        self.state.store(BATCH_READY, Ordering::Release);
    }

    fn mark_failed(&self, error: String) {
        *self
            .error
            .lock()
            .expect("poisoned BatchCompletion error lock") = Some(error);
        self.state.store(BATCH_FAILED, Ordering::Release);
    }

    pub fn wait(&self) {
        loop {
            match self.state.load(Ordering::Acquire) {
                BATCH_PENDING => std::hint::spin_loop(),
                BATCH_READY => return,
                BATCH_FAILED => {
                    let message = self
                        .error
                        .lock()
                        .expect("poisoned BatchCompletion error lock")
                        .clone()
                        .unwrap_or_else(|| "unknown ORT async failure".to_string());
                    eprintln!("RunAsync completion failed: {message}");
                    std::process::abort();
                }
                _ => unreachable!("invalid batch completion state"),
            }
        }
    }

    pub fn poll(&self) -> BatchPoll {
        match self.state.load(Ordering::Acquire) {
            BATCH_PENDING => BatchPoll::Pending,
            BATCH_READY => BatchPoll::Ready,
            BATCH_FAILED => BatchPoll::Failed,
            _ => unreachable!("invalid batch completion state"),
        }
    }
}

unsafe impl Send for BatchCompletion {}
unsafe impl Sync for BatchCompletion {}

struct BatchResources {
    _input_value: OrtValueHandle,
    _output_value: OrtValueHandle,
}

unsafe impl Send for BatchResources {}

/// One in-flight batch. Submission owns this until it reaches the batch queue;
/// completion owns it until the response bytes are encoded and the session buffer
/// can be reused.
pub struct InFlightBatch {
    pub completion: Arc<BatchCompletion>,
    pub output_ptr: *const f32,
    pub output_len: usize,
    pub session_available: Arc<AtomicBool>,
    /// Holds the moved request slices alive until completion. Submission drains
    /// them out of the ring so guard drop no longer determines their lifetime.
    pub input_slices: Vec<PoolSlice>,
    _resources: BatchResources,
}

unsafe impl Send for InFlightBatch {}

unsafe extern "system" fn run_async_callback(
    user_data: *mut c_void,
    _outputs: *mut *mut sys::OrtValue,
    _num_outputs: usize,
    status: sys::OrtStatusPtr,
) {
    let completion = unsafe { Arc::from_raw(user_data.cast::<BatchCompletion>()) };
    if status.0.is_null() {
        completion.mark_ready();
    } else {
        completion.mark_failed(status_message(status));
    }
    // `completion` is dropped here, releasing the callback's Arc clone.
}

/// Per-session state used by the submission thread.
pub struct InferenceSession {
    session: Session,
    input_memory_info: MemoryInfo,
    output_memory_info: MemoryInfo,
    _input_name: CString,
    output_name: CString,
    /// Stable `RunAsync` name arrays. The strings live in `input_name`/`output_name`.
    input_name_ptrs: [*const c_char; 1],
    output_name_ptrs: [*const c_char; 1],
    /// Stable `RunAsync` value arrays. Because each session is single-flight, these are
    /// only mutated while `available == true`, before submission, and after completion has
    /// retired the previous batch.
    input_value_ptrs: [*const sys::OrtValue; 1],
    output_value_ptrs: [*mut sys::OrtValue; 1],
    output_ptr: *mut f32,
    output_capacity: usize,
    available: Arc<AtomicBool>,
}

// SAFETY: InferenceSession is moved to its owning thread and never shared concurrently.
unsafe impl Send for InferenceSession {}

impl InferenceSession {
    /// Construct a session for the given ONNX model bytes.
    pub fn new(model_bytes: &[u8]) -> Self {
        Self::with_output_capacity(model_bytes, MAX_BATCH_VECTORS)
    }

    /// Construct a session with a caller-chosen maximum output vector capacity.
    pub fn with_output_capacity(model_bytes: &[u8], output_capacity: usize) -> Self {
        assert!(output_capacity > 0, "output_capacity must be > 0");
        #[allow(unused_mut)]
        let mut builder = Session::builder()
            .unwrap_or_else(|e| {
                eprintln!("Session::builder failed: {e}");
                std::process::abort()
            })
            .with_intra_threads(ORT_INTRA_THREADS)
            .unwrap_or_else(|e| {
                eprintln!("with_intra_threads failed: {e}");
                std::process::abort()
            });

        #[cfg(feature = "cuda")]
        let mut builder = builder
            .with_execution_providers([CUDAExecutionProvider::default().build()])
            .unwrap_or_else(|e| {
                eprintln!("with_execution_providers failed: {e}");
                std::process::abort()
            });

        let session = builder.commit_from_memory(model_bytes).unwrap_or_else(|e| {
            eprintln!("commit_from_memory failed: {e}");
            std::process::abort()
        });

        if session.inputs().len() != 1 || session.outputs().len() != 1 {
            eprintln!(
                "disrust: session expects exactly 1 input and 1 output, got {} inputs / {} outputs",
                session.inputs().len(),
                session.outputs().len()
            );
            std::process::abort();
        }

        let input_name = CString::new(session.inputs()[0].name()).unwrap_or_else(|e| {
            eprintln!("invalid input name: {e}");
            std::process::abort()
        });
        let output_name = CString::new(session.outputs()[0].name()).unwrap_or_else(|e| {
            eprintln!("invalid output name: {e}");
            std::process::abort()
        });

        #[cfg(feature = "cuda")]
        let input_memory_info = MemoryInfo::new(
            AllocationDevice::CUDA_PINNED,
            0,
            AllocatorType::Device,
            MemoryType::CPUInput,
        )
        .unwrap_or_else(|e| {
            eprintln!("input MemoryInfo::new failed: {e}");
            std::process::abort()
        });

        #[cfg(not(feature = "cuda"))]
        let input_memory_info = MemoryInfo::new(
            AllocationDevice::CPU,
            0,
            AllocatorType::Arena,
            MemoryType::Default,
        )
        .unwrap_or_else(|e| {
            eprintln!("input MemoryInfo::new failed: {e}");
            std::process::abort()
        });

        #[cfg(feature = "cuda")]
        let output_memory_info = MemoryInfo::new(
            AllocationDevice::CUDA_PINNED,
            0,
            AllocatorType::Device,
            MemoryType::CPUOutput,
        )
        .unwrap_or_else(|e| {
            eprintln!("output MemoryInfo::new failed: {e}");
            std::process::abort()
        });

        #[cfg(not(feature = "cuda"))]
        let output_memory_info = MemoryInfo::new(
            AllocationDevice::CPU,
            0,
            AllocatorType::Arena,
            MemoryType::Default,
        )
        .unwrap_or_else(|e| {
            eprintln!("output MemoryInfo::new failed: {e}");
            std::process::abort()
        });

        #[cfg(feature = "cuda")]
        let output_ptr = unsafe {
            let ptr =
                alloc_pinned(output_capacity * std::mem::size_of::<f32>()).unwrap_or_else(|e| {
                    eprintln!("{e}");
                    std::process::abort();
                });
            std::ptr::write_bytes(ptr, 0, output_capacity * std::mem::size_of::<f32>());
            ptr as *mut f32
        };

        #[cfg(not(feature = "cuda"))]
        let output_ptr = Box::leak(vec![0f32; output_capacity].into_boxed_slice()).as_mut_ptr();

        let input_name_ptrs = [input_name.as_ptr() as *const c_char];
        let output_name_ptrs = [output_name.as_ptr() as *const c_char];

        Self {
            session,
            input_memory_info,
            output_memory_info,
            _input_name: input_name,
            output_name,
            input_name_ptrs,
            output_name_ptrs,
            input_value_ptrs: [std::ptr::null()],
            output_value_ptrs: [std::ptr::null_mut()],
            output_ptr,
            output_capacity,
            available: Arc::new(AtomicBool::new(true)),
        }
    }

    pub fn try_acquire(&self) -> bool {
        self.available
            .compare_exchange(true, false, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    pub fn is_available(&self) -> bool {
        self.available.load(Ordering::Acquire)
    }

    pub fn output_name(&self) -> &str {
        self.output_name
            .to_str()
            .expect("ORT output name must be valid UTF-8")
    }

    pub fn submit_batch(
        &mut self,
        input_host_ptr: *const f32,
        num_vectors: usize,
    ) -> InFlightBatch {
        debug_assert!(num_vectors <= self.output_capacity);
        debug_assert!(
            !self.available.load(Ordering::Acquire),
            "submit_batch requires an acquired session"
        );

        let input_shape = [num_vectors as i64, FDIM as i64];
        let output_shape = [num_vectors as i64];
        let input_value = create_tensor_from_external(
            &self.input_memory_info,
            input_host_ptr as *mut c_void,
            num_vectors * FDIM,
            &input_shape,
        );
        let output_value = create_tensor_from_external(
            &self.output_memory_info,
            self.output_ptr.cast::<c_void>(),
            num_vectors,
            &output_shape,
        );

        let completion = Arc::new(BatchCompletion::new());
        let callback_completion = Arc::clone(&completion);
        let user_data = Arc::into_raw(callback_completion) as *mut c_void;

        self.input_value_ptrs[0] = input_value.as_ptr() as *const sys::OrtValue;
        self.output_value_ptrs[0] = output_value.as_ptr();

        // TODO: once the direct RunAsync path is stable, evaluate `disable_synchronize_execution_providers`
        // plus a user compute stream and explicit CUDA event synchronization. For now we rely on ORT's default
        // synchronization so callback completion implies the pinned output buffer is safe for CPU reads.
        let status = unsafe {
            (api().RunAsync)(
                self.session.ptr().cast_mut(),
                std::ptr::null(),
                self.input_name_ptrs.as_ptr(),
                self.input_value_ptrs.as_ptr(),
                self.input_value_ptrs.len(),
                self.output_name_ptrs.as_ptr(),
                self.output_name_ptrs.len(),
                self.output_value_ptrs.as_mut_ptr(),
                Some(run_async_callback),
                user_data,
            )
        };
        if !status.0.is_null() {
            unsafe { drop(Arc::from_raw(user_data.cast::<BatchCompletion>())) };
            check_status(status, "RunAsync failed");
        }

        // TODO: profile whether CUDA EP consumes CUDA_PINNED inputs via direct mapped reads or via internal async
        // staging copies. The structure here works either way, but the performance model differs.
        InFlightBatch {
            completion,
            output_ptr: self.output_ptr,
            output_len: num_vectors,
            session_available: Arc::clone(&self.available),
            input_slices: Vec::new(),
            _resources: BatchResources {
                _input_value: input_value,
                _output_value: output_value,
            },
        }
    }
}

impl Drop for InferenceSession {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        if let Err(e) = unsafe { free_pinned(self.output_ptr.cast::<c_void>()) } {
            eprintln!("InferenceSession drop failed to free pinned output buffer: {e}");
        }

        #[cfg(not(feature = "cuda"))]
        unsafe {
            drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
                self.output_ptr,
                self.output_capacity,
            )));
        }
    }
}
