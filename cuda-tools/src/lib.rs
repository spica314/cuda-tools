#![cfg_attr(target_arch = "nvptx64", no_std)]
#![feature(abi_ptx)]
#![feature(optin_builtin_traits)]

pub mod cuda_box;
pub mod cuda_slice;

#[cfg(not(target_arch = "nvptx64"))]
pub mod ffi;

#[cfg(not(target_arch = "nvptx64"))]
pub mod error;

#[cfg(not(target_arch = "nvptx64"))]
pub mod runtime;

#[macro_export]
macro_rules! include_kernel {
    ( ) => {
        include_str!(concat!(env!("OUT_DIR"), "/kernel.ptx"))
    };
}

#[macro_export]
macro_rules! build_kernel {
    ( $kernel_path:expr, $kernel_crate_name:expr ) => {
        use std::io::Write;
        use std::process::Command;
        let output = Command::new(env!("CARGO"))
            .args(&[
                "rustc",
                "--lib",
                "--release",
                "--target",
                "nvptx64-nvidia-cuda",
                "--",
                "--emit",
                "asm",
            ])
            .current_dir($kernel_path)
            .output()
            .unwrap();
        println!("status: {}", output.status);
        println!("warning={}", std::str::from_utf8(&output.stdout).unwrap());
        println!("warning={}", std::str::from_utf8(&output.stderr).unwrap());
        if !output.status.success() {
            panic!();
        }
        let crate_name = $kernel_crate_name.to_string().replace("-", "_");
        let crate_dir = env!("CARGO_MANIFEST_DIR");
        let query = format!(
            "{}/{}/target/nvptx64-nvidia-cuda/release/deps/{}*.s",
            crate_dir, $kernel_path, crate_name
        )
        .to_string();
        for entry in glob::glob(&query).expect("") {
            match entry {
                Ok(path) => {
                    let origin = std::fs::read_to_string(path.to_str().unwrap()).unwrap();
                    let mut out_file = std::fs::File::create(&format!(
                        "{}/kernel.ptx",
                        std::env::var("OUT_DIR").unwrap()
                    ))
                    .unwrap();
                    let mut writer = std::io::BufWriter::new(out_file);
                    writer.write_all(origin.as_bytes()).unwrap();
                }
                Err(e) => eprintln!("{:?}", e),
            }
        }
        let query = format!("{}/{}/**/*.rs", crate_dir, $kernel_path).to_string();
        for entry in glob::glob(&query).expect("") {
            match entry {
                Ok(path) => {
                    println!("cargo:rerun-if-changed={}", path.to_str().unwrap());
                    path.to_str().unwrap();
                }
                Err(e) => eprintln!("{:?}", e),
            }
        }
        println!("cargo:rerun-if-changed=build.rs");
    };
}
