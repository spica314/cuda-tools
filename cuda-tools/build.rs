use std::env;
use std::path::PathBuf;

fn main() {
    let cuda_path = env::var("CUDA_PATH").expect("env CUDA_PATH is empty");
    println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=stdc++");
    let bindings = bindgen::Builder::default()
        .header("cuda_headers.h")
        .clang_arg(&format!("-I{}/include", cuda_path))
        .rustified_enum("cudaError_enum")
        .rustified_enum("CUctx_flags_enum")
        .whitelist_var("cu.*")
        .whitelist_var("CU.*")
        .whitelist_type("cu.*")
        .whitelist_type("CU.*")
        .whitelist_function("cu.*")
        .generate()
        .expect("Unable to generate bindings");
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
