#[cfg(not(target_arch = "nvptx64"))]
use std::prelude::v1::*;

use crate::ffi;

#[derive(Debug)]
pub enum CUDAError {
    CUResult(ffi::cudaError_enum),
    DeviceIdIsOutOfRange,
}

impl CUDAError {
    pub fn new(x: ffi::cudaError_enum) -> CUDAError {
        CUDAError::CUResult(x)
    }
}
