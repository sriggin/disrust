//! Inference backend trait and ORT session implementation.

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

use crate::buffer_pool::{BufferPool, PoolSlice};
#[cfg(feature = "cuda")]
use crate::config::GPU_BUFFER_POOL_BYTES;
use crate::config::GPU_BUFFER_POOL_CAPACITY;
use crate::config::{MAX_BATCH_VECTORS, ORT_INTRA_THREADS};
use crate::constants::FEATURE_DIM as FDIM;

#[cfg(feature = "cuda")]
use crate::cuda::memory::{alloc_pinned, free_pinned};

/// Pluggable inference backend.
///
/// Implementors provide a pool of single-flight sessions. The server calls
/// `try_acquire` / `is_available` to manage the round-robin pool, then
/// `submit_batch` to dispatch a contiguous slice of feature vectors and
/// receive an `InFlightBatch` that signals completion asynchronously.
///
/// `Resources` is the backend-specific state that must stay alive until the
/// batch completes (e.g. ORT tensor handles). It is stored inline in
/// `InFlightBatch<Self::Resources>` — no heap allocation is incurred.
///
/// Startup sequence: `B::init()` → `B::make_pool()` → construct `Vec<B>`.
pub trait InferenceBackend: Send {
    type Resources: Send;

    /// One-time global initialization for this backend, run before `make_pool`
    /// or any session construction. Backends that require runtime library
    /// loading, GPU driver checks, or global state setup implement this.
    /// The default is a no-op.
    fn init()
    where
        Self: Sized,
    {
    }

    /// Allocate and leak the server's input feature pool. Called once at
    /// startup before sessions are constructed. The backend decides the memory
    /// type: pinned host memory for zero-copy GPU DMA, or regular heap.
    ///
    /// The returned reference is `'static` because the pool lives for the
    /// process lifetime.
    fn make_pool() -> &'static BufferPool
    where
        Self: Sized;

    fn try_acquire(&mut self) -> bool;
    fn is_available(&self) -> bool;

    /// Submit a batch for inference. `input_host_ptr` points to a contiguous
    /// row-major `[num_vectors × FEATURE_DIM]` f32 array in the pool returned
    /// by `make_pool`. Returns an `InFlightBatch` whose `completion` will be
    /// signaled when `output_ptr` is safe to read.
    fn submit_batch(
        &mut self,
        input_host_ptr: *const f32,
        num_vectors: usize,
    ) -> InFlightBatch<Self::Resources>;
}

// ---------------------------------------------------------------------------
// BatchCompletion
// ---------------------------------------------------------------------------

const BATCH_PENDING: u8 = 0;
const BATCH_READY: u8 = 1;
const BATCH_FAILED: u8 = 2;

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

impl Default for BatchCompletion {
    fn default() -> Self {
        Self::new()
    }
}

impl BatchCompletion {
    pub fn new() -> Self {
        Self {
            state: AtomicU8::new(BATCH_PENDING),
            error: Mutex::new(None),
        }
    }

    pub fn mark_ready(&self) {
        self.state.store(BATCH_READY, Ordering::Release);
    }

    pub fn mark_failed(&self, error: String) {
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
                        .unwrap_or_else(|| "unknown async inference failure".to_string());
                    eprintln!("batch completion failed: {message}");
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

// ---------------------------------------------------------------------------
// InFlightBatch
// ---------------------------------------------------------------------------

/// One in-flight batch. Submission owns this until it reaches the inflight
/// queue; completion owns it until the response bytes are encoded and the
/// session buffer can be reused.
///
/// `R` is the backend's resource type, stored inline to avoid heap allocation.
pub struct InFlightBatch<R: Send> {
    pub completion: Arc<BatchCompletion>,
    pub output_ptr: *const f32,
    pub output_len: usize,
    pub session_available: Arc<AtomicBool>,
    /// Holds the moved request slices alive until completion. Submission drains
    /// them out of the ring so guard drop no longer determines their lifetime.
    pub input_slices: Vec<PoolSlice>,
    /// Backend-specific resources that must stay alive until completion
    /// (e.g. ORT tensor handles, XGBoost DMatrix). Stored inline — no allocation.
    _resources: R,
}

// SAFETY: output_ptr is a raw pointer into a pinned buffer owned by the session.
// The session remains alive (and the buffer valid) until session_available is
// reset to true, which only happens after completion drops InFlightBatch.
unsafe impl<R: Send> Send for InFlightBatch<R> {}

impl<R: Send> InFlightBatch<R> {
    pub(crate) fn new(
        completion: Arc<BatchCompletion>,
        output_ptr: *const f32,
        output_len: usize,
        session_available: Arc<AtomicBool>,
        resources: R,
    ) -> Self {
        Self {
            completion,
            output_ptr,
            output_len,
            session_available,
            input_slices: Vec::new(),
            _resources: resources,
        }
    }
}

// ---------------------------------------------------------------------------
// OrtSession — InferenceBackend implementation via ORT RunAsync
// ---------------------------------------------------------------------------

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

/// ORT tensor handles that must stay alive until the RunAsync callback fires.
pub struct OrtBatchResources {
    _input_value: OrtValueHandle,
    _output_value: OrtValueHandle,
}

unsafe impl Send for OrtBatchResources {}

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

/// Per-session ORT state used by the inference thread.
pub struct OrtSession {
    session: Session,
    input_memory_info: MemoryInfo,
    output_memory_info: MemoryInfo,
    _input_name: CString,
    output_name: CString,
    /// Stable `RunAsync` name arrays. The strings live in `_input_name`/`output_name`.
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

// SAFETY: OrtSession is moved to its owning thread and never shared concurrently.
unsafe impl Send for OrtSession {}

impl OrtSession {
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

    pub fn output_name(&self) -> &str {
        self.output_name
            .to_str()
            .expect("ORT output name must be valid UTF-8")
    }
}

impl OrtSession {
    pub(crate) fn try_acquire(&self) -> bool {
        self.available
            .compare_exchange(true, false, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    pub(crate) fn is_available(&self) -> bool {
        self.available.load(Ordering::Acquire)
    }

    pub(crate) fn do_submit_batch(
        &mut self,
        input_host_ptr: *const f32,
        num_vectors: usize,
    ) -> InFlightBatch<OrtBatchResources> {
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
        InFlightBatch::new(
            completion,
            self.output_ptr,
            num_vectors,
            Arc::clone(&self.available),
            OrtBatchResources {
                _input_value: input_value,
                _output_value: output_value,
            },
        )
    }
}

impl Drop for OrtSession {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        if let Err(e) = unsafe { free_pinned(self.output_ptr.cast::<c_void>()) } {
            eprintln!("OrtSession drop failed to free pinned output buffer: {e}");
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

// ---------------------------------------------------------------------------
// OrtBackend — pool of OrtSession slots; implements InferenceBackend
// ---------------------------------------------------------------------------

/// The ORT inference backend. Owns a pool of single-flight `OrtSession` slots
/// and manages round-robin acquisition. This is the type passed to
/// `InferenceConsumer<B>`.
pub struct OrtBackend {
    sessions: Vec<OrtSession>,
    cursor: usize,
    /// Index of the slot claimed by the most recent successful `try_acquire`.
    /// Consumed by the next `submit_batch` call.
    acquired: Option<usize>,
}

// SAFETY: OrtBackend is moved to its owning thread and never shared concurrently.
unsafe impl Send for OrtBackend {}

impl OrtBackend {
    pub fn new(model_bytes: &[u8], pool_size: usize) -> Self {
        assert!(pool_size > 0, "pool_size must be > 0");
        let sessions = (0..pool_size)
            .map(|_| OrtSession::new(model_bytes))
            .collect();
        Self {
            sessions,
            cursor: 0,
            acquired: None,
        }
    }

    /// Construct a backend with a single slot of the given output capacity.
    /// Used by the verify path for targeted single-batch testing.
    pub fn new_with_capacity(model_bytes: &[u8], output_capacity: usize) -> Self {
        assert!(output_capacity > 0, "output_capacity must be > 0");
        let sessions = vec![OrtSession::with_output_capacity(
            model_bytes,
            output_capacity,
        )];
        Self {
            sessions,
            cursor: 0,
            acquired: None,
        }
    }

    /// Name of the model's output tensor. Delegates to the first slot.
    pub fn output_name(&self) -> &str {
        self.sessions[0].output_name()
    }
}

impl InferenceBackend for OrtBackend {
    type Resources = OrtBatchResources;

    fn init() {
        #[cfg(feature = "cuda")]
        {
            crate::cuda::preflight::verify_cuda_startup().unwrap_or_else(|e| {
                eprintln!("disrust preflight failed: {e}");
                std::process::exit(1);
            });
            eprintln!("disrust: CUDA driver preflight ok");
        }
    }

    fn make_pool() -> &'static BufferPool {
        let ptr: *mut f32 = {
            #[cfg(feature = "cuda")]
            {
                unsafe {
                    let mut raw: *mut std::ffi::c_void = std::ptr::null_mut();
                    let status =
                        cudarc::driver::sys::cuMemAllocHost_v2(&mut raw, GPU_BUFFER_POOL_BYTES);
                    if status != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                        eprintln!("OrtBackend::make_pool: cuMemAllocHost_v2 failed: {status:?}");
                        std::process::abort();
                    }
                    std::ptr::write_bytes(raw as *mut u8, 0u8, GPU_BUFFER_POOL_BYTES);
                    raw as *mut f32
                }
            }

            #[cfg(not(feature = "cuda"))]
            {
                Box::leak(vec![0f32; GPU_BUFFER_POOL_CAPACITY].into_boxed_slice()).as_mut_ptr()
            }
        };

        Box::leak(unsafe { BufferPool::from_raw_ptr(ptr, GPU_BUFFER_POOL_CAPACITY) })
    }

    fn is_available(&self) -> bool {
        self.sessions.iter().any(|s| s.is_available())
    }

    fn try_acquire(&mut self) -> bool {
        for _ in 0..self.sessions.len() {
            let idx = self.cursor;
            self.cursor = (idx + 1) % self.sessions.len();
            if self.sessions[idx].try_acquire() {
                self.acquired = Some(idx);
                return true;
            }
        }
        false
    }

    fn submit_batch(
        &mut self,
        input_host_ptr: *const f32,
        num_vectors: usize,
    ) -> InFlightBatch<OrtBatchResources> {
        let idx = self
            .acquired
            .take()
            .expect("submit_batch called without a prior successful try_acquire");
        self.sessions[idx].do_submit_batch(input_host_ptr, num_vectors)
    }
}
