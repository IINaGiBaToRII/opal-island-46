
pub const QOS_KEEP: u8 = 0;
pub const QOS_BACKGROUND: u8 = 1;
pub const QOS_UTILITY: u8 = 2;
pub const QOS_DEFAULT: u8 = 3;
pub const QOS_USER_INITIATED: u8 = 4;

#[cfg(target_os = "macos")]
pub fn apply(level: u8) -> bool {
    let class = match level {
        QOS_BACKGROUND => libc::qos_class_t::QOS_CLASS_BACKGROUND,
        QOS_UTILITY => libc::qos_class_t::QOS_CLASS_UTILITY,
        QOS_DEFAULT => libc::qos_class_t::QOS_CLASS_DEFAULT,
        QOS_USER_INITIATED => libc::qos_class_t::QOS_CLASS_USER_INITIATED,
        _ => return false,
    };
    unsafe { libc::pthread_set_qos_class_self_np(class, 0) == 0 }
}

#[cfg(not(target_os = "macos"))]
pub fn apply(_level: u8) -> bool {
    false
}

#[cfg(target_os = "macos")]
pub fn current() -> u8 {
    let mut class = libc::qos_class_t::QOS_CLASS_UNSPECIFIED;
    let mut prio: libc::c_int = 0;
    let r = unsafe { libc::pthread_get_qos_class_np(libc::pthread_self(), &mut class, &mut prio) };
    if r != 0 {
        return 0;
    }
    match class {
        libc::qos_class_t::QOS_CLASS_BACKGROUND => QOS_BACKGROUND,
        libc::qos_class_t::QOS_CLASS_UTILITY => QOS_UTILITY,
        libc::qos_class_t::QOS_CLASS_DEFAULT => QOS_DEFAULT,
        libc::qos_class_t::QOS_CLASS_USER_INITIATED => QOS_USER_INITIATED,
        _ => 0,
    }
}

#[cfg(not(target_os = "macos"))]
pub fn current() -> u8 {
    0
}
