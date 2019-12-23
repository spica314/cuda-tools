#[cfg(not(target_arch = "nvptx64"))]
use crate::error::*;
#[cfg(not(target_arch = "nvptx64"))]
use crate::ffi;
use core::ops::Deref;

// (global) memory on device
pub struct CUDABox<'a, T> {
    ptr: &'a T,
}

impl<'a, T> Deref for CUDABox<'a, T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        self.ptr
    }
}

impl<'a, T> CUDABox<'a, T> {
    #[cfg(not(target_arch = "nvptx64"))]
    pub fn new(ptr: &'a T) -> CUDABox<'a, T> {
        CUDABox { ptr }
    }

    // get reference
    pub fn get(&self) -> &T {
        self.ptr
    }

    // get host T
    #[cfg(not(target_arch = "nvptx64"))]
    pub fn to_host(&self) -> Result<T, CUDAError> {
        let mut res = unsafe { core::mem::MaybeUninit::zeroed().assume_init() };
        let result = unsafe {
            ffi::cuMemcpyDtoH_v2(
                &mut res as *mut T as *mut std::os::raw::c_void,
                self.ptr as *const T as ffi::DevicePtr,
                std::mem::size_of::<T>(),
            )
        };
        if result != ffi::CUresult::CUDA_SUCCESS {
            return Err(CUDAError::new(result));
        }
        Ok(res)
    }
}
