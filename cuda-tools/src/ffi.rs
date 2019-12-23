#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

pub type DevicePtr = u64;
#[cfg(target_arch = "x86_64")]
#[test]
fn test_device_ptr_size() {
    assert_eq!(
        core::mem::size_of::<CUdeviceptr>(),
        core::mem::size_of::<u64>()
    );
}
