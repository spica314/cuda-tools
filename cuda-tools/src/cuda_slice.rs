#[cfg(not(target_arch = "nvptx64"))]
use crate::error::*;
#[cfg(not(target_arch = "nvptx64"))]
use crate::ffi;
#[cfg(target_arch = "nvptx64")]
use core::ops::Deref;

// (global) memory on device
pub struct CUDASlice<'a, T> {
    ptr: &'a [T],
}

#[cfg(target_arch = "nvptx64")]
impl<'a, T> Deref for CUDASlice<'a, T> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        self.ptr
    }
}

impl<'a, T> CUDASlice<'a, T> {
    #[cfg(not(target_arch = "nvptx64"))]
    pub fn new(ptr: ffi::DevicePtr, len: usize) -> CUDASlice<'a, T> {
        unsafe {
            CUDASlice {
                ptr: core::slice::from_raw_parts(ptr as *const T, len),
            }
        }
    }

    // get reference
    #[cfg(target_arch = "nvptx64")]
    pub fn get(&self) -> &[T] {
        self.ptr
    }

    // get host T
    #[cfg(not(target_arch = "nvptx64"))]
    pub fn to_host(&self) -> Result<Vec<T>, CUDAError> {
        let mut res = vec![];
        unsafe {
            for _ in 0..self.ptr.len() {
                res.push(core::mem::MaybeUninit::zeroed().assume_init());
            }
        }
        let result = unsafe {
            ffi::cuMemcpyDtoH_v2(
                res.as_mut_ptr() as *mut std::os::raw::c_void,
                self.ptr.as_ptr() as ffi::DevicePtr,
                std::mem::size_of::<T>() * self.ptr.len(),
            )
        };
        if result != ffi::CUresult::CUDA_SUCCESS {
            return Err(CUDAError::new(result));
        }
        Ok(res)
    }
}
