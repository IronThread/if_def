use dirs::{cache_dir as temp_dir, home_dir};
use std::borrow::Cow;
use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::prelude::*;
use std::path::PathBuf;
use std::process::{exit, Command};

fn main() -> Result<(), Box<dyn Error>> {
    if fs::metadata("src/consts.rs").is_ok() {
        return Ok(());
    }

    if let Some(e) = Command::new("rustup")
        .args(&["component", "add", "rust-src"][..])
        .status()?
        .code()
    {
        if e > 0 {
            exit(e)
        }
    }

    let output = Command::new("rustup").arg("default").output()?;

    if let Some(e) = output.status.code() {
        if e > 0 {
            exit(e)
        }
    }

    let mut version_ = String::from_utf8(output.stdout)?;

    let i = version_.find(' ').unwrap_or(version_.len());

    let version = &version_[..i];

    let mut lib = home_dir().unwrap();

    lib.push(".rustup/toolchains");
    lib.push(version);
    lib.push("share/doc/rust/html");

    println!("cargo:rerun-if-changed={}", lib.display());

    lib.push("core");

    let mut f = File::create("src/consts.rs")?;
    writeln!(f, r#"const CORE_PATH: &str = r"{}";"#, lib.display())?;

    lib.pop();
    lib.push("std");
    writeln!(f, r#"const STD_PATH: &str = r"{}";"#, lib.display())?;

    lib.pop();
    lib.push("alloc");
    writeln!(f, r#"const ALLOC_PATH: &str = r"{}";"#, lib.display())?;

    lib.pop();
    lib.push("proc_macro");
    writeln!(f, r#"const PROC_MACRO_PATH: &str = r"{}";"#, lib.display())?;

    lib.pop();
    lib.push("test");
    writeln!(f, r#"const TEST_PATH: &str = r"{}";"#, lib.display())?;

    for _ in 0..7 {
        lib.pop();
    }

    lib.push(".cargo/registry/src");
    writeln!(
        f,
        r#"const SRC_REGISTRY_PATH: &str = r"{}";"#,
        lib.display()
    )?;

    let cargo_home = std::env::var("CARGO_HOME").unwrap();
    let home = cargo_home
        .strip_suffix(&['/', '\\'][..])
        .unwrap_or(&cargo_home);
    writeln!(
        f,
        r#"const CARGO_CACHE: &str = r"{}/registry/cache";"#,
        home
    )?;
    writeln!(
        f,
        r#"const CARGO_PACKAGE_CACHE: &str = r"{}/.package-cache";"#,
        home
    )?;

    Ok(())
}
