//! Inference pipeline: always-compiled session management, submission, and completion.

pub mod batch_queue;
pub mod completion;
pub mod connection_registry;
pub mod inference;
pub mod ready_queue;
pub mod session;
pub mod submission;
pub mod writer;

pub use session::{InferenceBackend, OrtBackend};

use std::path::PathBuf;

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
