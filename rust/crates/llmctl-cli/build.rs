//! Build identity for `llmctl-next build-info` (M7c). The release script
//! sets LLMCTL_GIT_SHA; a dev build reports "dev". `--version` stays one
//! line (`llmctl-next <version>`): `host add` parses its last word.

fn main() {
    println!("cargo:rerun-if-env-changed=LLMCTL_GIT_SHA");
    println!("cargo:rerun-if-env-changed=LLMCTL_BUILD_HOST");
    let sha = std::env::var("LLMCTL_GIT_SHA").unwrap_or_else(|_| "dev".into());
    let host = std::env::var("LLMCTL_BUILD_HOST").unwrap_or_default();
    let target = std::env::var("TARGET").unwrap_or_default();
    let profile = std::env::var("PROFILE").unwrap_or_default();
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let rustc_v = std::process::Command::new(rustc)
        .arg("-V")
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    println!("cargo:rustc-env=LLMCTL_BUILD_GIT_SHA={sha}");
    println!("cargo:rustc-env=LLMCTL_BUILD_HOST={host}");
    println!("cargo:rustc-env=LLMCTL_BUILD_TARGET={target}");
    println!("cargo:rustc-env=LLMCTL_BUILD_PROFILE={profile}");
    println!("cargo:rustc-env=LLMCTL_BUILD_RUSTC={rustc_v}");
}
