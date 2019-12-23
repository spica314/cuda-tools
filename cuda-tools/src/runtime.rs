#[cfg(not(target_arch = "nvptx64"))]
use std::prelude::v1::*;

use crate::error::*;

use crate::ffi::{self, DevicePtr};

use crate::cuda_box::*;
use crate::cuda_slice::*;

use std::collections::BTreeMap;

static mut CUDA_INITIALIZED: bool = false;
pub struct Runtime {
    context: ffi::CUcontext,
    module: ffi::CUmodule,
    function_names: BTreeMap<*mut core::ffi::c_void, String>,
}

impl Runtime {
    pub fn new(device_id: i32, kernel_str: &str) -> Result<Runtime, CUDAError> {
        // cuInit
        unsafe {
            if !CUDA_INITIALIZED {
                let result = ffi::cuInit(0);
                if result != ffi::CUresult::CUDA_SUCCESS {
                    return Err(CUDAError::new(result));
                }
                CUDA_INITIALIZED = true;
            }
        }
        // device check
        let mut device_num: i32 = 0;
        let result = unsafe { ffi::cuDeviceGetCount(&mut device_num as *mut i32) };
        if result != ffi::CUresult::CUDA_SUCCESS {
            return Err(CUDAError::new(result));
        }
        if device_id >= device_num {
            return Err(CUDAError::DeviceIdIsOutOfRange);
        }
        // context
        let mut context: ffi::CUcontext = unsafe { core::mem::MaybeUninit::zeroed().assume_init() };
        let result = unsafe {
            ffi::cuCtxCreate_v2(
                &mut context as *mut ffi::CUcontext,
                ffi::CUctx_flags_enum::CU_CTX_SCHED_AUTO as u32,
                device_id,
            )
        };
        if result != ffi::CUresult::CUDA_SUCCESS {
            return Err(CUDAError::new(result));
        }
        // module
        let mut module: ffi::CUmodule = unsafe { core::mem::MaybeUninit::zeroed().assume_init() };
        let kernel_cstr = std::ffi::CString::new(kernel_str).unwrap();
        let result = unsafe {
            ffi::cuModuleLoadData(
                &mut module as *mut ffi::CUmodule,
                kernel_cstr.as_ptr() as *mut std::ffi::c_void,
            )
        };
        if result != ffi::CUresult::CUDA_SUCCESS {
            return Err(CUDAError::new(result));
        }
        // res
        let res = Runtime {
            context,
            module,
            function_names: BTreeMap::new(),
        };
        Ok(res)
    }
    pub fn record_function_name<Args>(&mut self, f: extern "ptx-kernel" fn(&Args), name: &str) {
        self.function_names
            .insert(f as *mut core::ffi::c_void, name.to_string());
    }
    pub fn record_function_with_shared_memory_name<Args, SharedMemory>(
        &mut self,
        f: extern "ptx-kernel" fn(&Args, &SharedMemory),
        name: &str,
    ) {
        self.function_names
            .insert(f as *mut core::ffi::c_void, name.to_string());
    }
    pub fn launch<Args>(
        &self,
        f: extern "ptx-kernel" fn(&Args),
        args: &Args,
        grid_dim_x: usize,
        grid_dim_y: usize,
        grid_dim_z: usize,
        block_dim_x: usize,
        block_dim_y: usize,
        block_dim_z: usize,
    ) -> Result<(), CUDAError> {
        // function
        let func_name = self
            .function_names
            .get(&(f as *mut core::ffi::c_void))
            .unwrap();
        let mut function: ffi::CUfunction =
            unsafe { core::mem::MaybeUninit::zeroed().assume_init() };
        let func_name_cstr = std::ffi::CString::new(func_name.as_str()).unwrap();
        let result = unsafe {
            ffi::cuModuleGetFunction(
                &mut function as *mut ffi::CUfunction,
                self.module,
                func_name_cstr.as_ptr(),
            )
        };
        if result != ffi::CUresult::CUDA_SUCCESS {
            return Err(CUDAError::new(result));
        }
        // launch
        let args_d = self.alloc(args).unwrap();
        let mut args_d_ptr = args_d.get();
        let mut launch_args = vec![&mut args_d_ptr as *mut &Args as *mut std::ffi::c_void];
        unsafe {
            use std::os::raw::c_uint;
            use std::ptr;
            let shared_mem_bytes = 0;
            let result = ffi::cuLaunchKernel(
                function,
                grid_dim_x as c_uint,
                grid_dim_y as c_uint,
                grid_dim_z as c_uint,
                block_dim_x as c_uint,
                block_dim_y as c_uint,
                block_dim_z as c_uint,
                shared_mem_bytes as c_uint,
                ptr::null_mut(),
                launch_args.as_mut_ptr(),
                ptr::null_mut(),
            );
            if result != ffi::CUresult::CUDA_SUCCESS {
                return Err(CUDAError::new(result));
            }
        }
        Ok(())
    }
    pub fn alloc<'a, T>(&'a self, x: &T) -> Result<CUDABox<'a, T>, CUDAError> {
        // allocate
        let mut ptr: DevicePtr = 0;
        let size = std::mem::size_of::<T>();
        let result = unsafe { ffi::cuMemAlloc_v2(&mut ptr as *mut u64, size) };
        if result != ffi::CUresult::CUDA_SUCCESS {
            eprintln!("bug-a");
            return Err(CUDAError::new(result));
        }
        // memcpy
        let result = unsafe {
            ffi::cuMemcpyHtoD_v2(
                ptr,
                x as *const T as *const std::ffi::c_void,
                std::mem::size_of::<T>(),
            )
        };
        if result != ffi::CUresult::CUDA_SUCCESS {
            eprintln!("bug-b");
            return Err(CUDAError::new(result));
        }
        // return
        let ptr = unsafe { (ptr as *mut T).as_mut().unwrap() };
        Ok(CUDABox::new(ptr))
    }
    pub fn alloc_slice<'a, T>(&'a self, xs: &[T]) -> Result<CUDASlice<'a, T>, CUDAError> {
        // allocate
        let mut ptr: DevicePtr = 0;
        let size = std::mem::size_of::<T>() * xs.len();
        let result = unsafe { ffi::cuMemAlloc_v2(&mut ptr as *mut u64, size) };
        if result != ffi::CUresult::CUDA_SUCCESS {
            eprintln!("bug-a");
            return Err(CUDAError::new(result));
        }
        // memcpy
        let result = unsafe {
            ffi::cuMemcpyHtoD_v2(
                ptr,
                xs.as_ptr() as *const T as *const std::ffi::c_void,
                std::mem::size_of::<T>() * xs.len(),
            )
        };
        if result != ffi::CUresult::CUDA_SUCCESS {
            eprintln!("bug-b");
            return Err(CUDAError::new(result));
        }
        // return
        Ok(CUDASlice::new(ptr, xs.len()))
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        unsafe {
            ffi::cuCtxDestroy_v2(self.context);
        }
    }
}
