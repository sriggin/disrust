use std::io;

#[cfg(target_os = "linux")]
pub fn pin_current_thread(cpu: usize, thread_name: &str) -> io::Result<()> {
    // Linux treats pid 0 here as "current thread".
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        libc::CPU_SET(cpu, &mut set);
        let rc = libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
        if rc == 0 {
            Ok(())
        } else {
            Err(io::Error::new(
                io::Error::last_os_error().kind(),
                format!(
                    "failed to pin thread '{thread_name}' to CPU {cpu}: {}",
                    io::Error::last_os_error()
                ),
            ))
        }
    }
}

#[cfg(not(target_os = "linux"))]
pub fn pin_current_thread(cpu: usize, thread_name: &str) -> io::Result<()> {
    let _ = cpu;
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        format!("thread affinity for '{thread_name}' is only supported on Linux"),
    ))
}
