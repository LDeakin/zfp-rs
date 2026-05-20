use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Command;

#[test]
fn c_code_can_link_either_zfp_backend() {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("zfp-rs-ffi is inside the workspace");
    let cmake_source = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("c-dropin");
    let root = workspace.join("target").join("zfp-rs-ffi-c-dropin");
    let rust_target = root.join("cargo-target");

    build_rust_staticlib(workspace, &rust_target);

    run_cmake_backend(
        "c",
        &cmake_source,
        &root.join("cmake-c"),
        [
            cmake_arg("ZFP_DROPIN_BACKEND", "c"),
            cmake_arg("ZFP_SOURCE_DIR", workspace.join("zfp")),
        ],
    );

    run_cmake_backend(
        "rust",
        &cmake_source,
        &root.join("cmake-rust"),
        [
            cmake_arg("ZFP_DROPIN_BACKEND", "rust"),
            cmake_arg("ZFP_RS_FFI_LIBRARY", staticlib_path(&rust_target)),
            cmake_arg(
                "ZFP_INCLUDE_DIR",
                workspace.join("zfp-rs-ffi").join("include"),
            ),
        ],
    );
}

fn build_rust_staticlib(workspace: &Path, target_dir: &Path) {
    run(
        Command::new(cargo())
            .current_dir(workspace)
            .arg("build")
            .arg("-p")
            .arg("zfp-rs-ffi")
            .arg("--target-dir")
            .arg(target_dir),
        "building zfp-rs-ffi staticlib",
    );
}

fn run_cmake_backend<const N: usize>(
    backend: &str,
    source: &Path,
    build_dir: &Path,
    args: [String; N],
) {
    let mut configure = Command::new("cmake");
    configure.arg("-S").arg(source).arg("-B").arg(build_dir);
    for arg in args {
        configure.arg(arg);
    }
    run(
        &mut configure,
        &format!("configuring C drop-in test for {backend} backend"),
    );

    run(
        Command::new("cmake")
            .arg("--build")
            .arg(build_dir)
            .arg("--config")
            .arg("Debug"),
        &format!("building C drop-in test for {backend} backend"),
    );

    run(
        Command::new("ctest")
            .arg("--test-dir")
            .arg(build_dir)
            .arg("--output-on-failure")
            .arg("-C")
            .arg("Debug"),
        &format!("running C drop-in test for {backend} backend"),
    );
}

fn run(command: &mut Command, description: &str) {
    let output = command
        .output()
        .unwrap_or_else(|err| panic!("failed {description}: {err}"));
    if !output.status.success() {
        panic!(
            "failed {description}\nstatus: {}\nstdout:\n{}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

fn cargo() -> impl AsRef<OsStr> {
    std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into())
}

fn cmake_arg(name: &str, value: impl AsRef<Path>) -> String {
    format!("-D{name}={}", value.as_ref().display())
}

fn staticlib_path(target_dir: &Path) -> PathBuf {
    let filename = if cfg!(target_os = "windows") {
        "zfp_rs_ffi.lib"
    } else {
        "libzfp_rs_ffi.a"
    };
    target_dir.join("debug").join(filename)
}
