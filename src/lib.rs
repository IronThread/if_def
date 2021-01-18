#![feature(proc_macro_span, thread_spawn_unchecked, proc_macro_quote)]

use ::{
    dirs::{cache_dir as temp_dir, home_dir},
    proc_macro::TokenStream,
    quote::ToTokens,
    rand::prelude::*,
    std::{
        borrow::Cow,
        collections::HashMap,
        env,
        fs::{self, canonicalize, File},
        io::{self, prelude::*, SeekFrom},
        iter, mem,
        path::{Path, PathBuf},
        process::Command,
    },
    sync_2::Mutex,
};

static TEMP_DIR: Mutex<Option<PathBuf>> = Mutex::new(None);
static CRATE_DIR: Mutex<Option<PathBuf>> = Mutex::new(None);

// bitbuf it's a crate of my own that got the convenience of having a short name and the fact will
// never have the item it's taking
const RESERVED_PATH: &'static str = "::bitbuf::a";

fn if_def_internal(input2: syn::Path) -> bool {
    let span = input2
        .segments
        .first()
        .expect("empty path")
        .ident
        .span()
        .unwrap();
    let input2 = input2.into_token_stream().to_string();

    if input2 == RESERVED_PATH {
        return false;
    }

    let import = format!("use {} as _;", input2);

    let mut t = TEMP_DIR.lock().unwrap();
    let mut temp_dir = t.get_or_insert_with(|| {
        let mut p = temp_dir()
            .or_else(|| env::var("OUT_DIR").ok().map(PathBuf::from))
            .unwrap_or_default();

        p.push("rust_if_def");
        p
    });

    let mut ctd = CRATE_DIR.lock().unwrap();
    let mut crate_dir = ctd.get_or_insert_with(|| {
        env::var_os("CARGO_MANIFEST_DIR")
            .map(PathBuf::from)
            .unwrap_or_default()
    });

    let mut start = span.start();
    let mut end = span.end();
    let p = span.source_file().path();

    let file = p
        .file_name()
        .expect("maybe not that real file lacks filename");

    let rand_int: u128 = random();
    let crate_n = rand_int.to_string();

    let mut buffer = String::new();
    let mut cr = false;
    let mut start_index = 0;
    let mut last_opened = None;

    temp_dir.push(&crate_n);

    fn copy_all<T: AsRef<Path>, U: AsRef<Path>>(
        src: T,
        mut temp_dir: &mut PathBuf,
        last_opened: &mut Option<(File, File)>,
        buffer: &mut String,
        file: U,
    ) {
        for e in fs::read_dir(src).expect("failed to read src") {
            let entry = e.expect("failed to get entry of src");
            let path = entry.path();
            let file_name = path.file_name().unwrap();
            temp_dir.push(file_name);

            if entry.metadata().unwrap().is_dir() {
                fs::create_dir(&temp_dir);
                copy_all(
                    &*path,
                    &mut *temp_dir,
                    last_opened,
                    buffer,
                    file.as_ref(),
                );
            } else {
                let mut f =
                    File::create(&*temp_dir).expect("failed to copy from src,creating a file.");
                let mut r = File::open(&*path).expect("failed to copy from src,reading a file.");

                if last_opened.is_none() && file_name == file.as_ref().as_os_str() {
                    *last_opened = Some((r, f));
                } else {
                    unsafe {
                        let buffer = buffer.as_mut_vec();
                        r.read_to_end(buffer);
                        f.write_all(&buffer[..]);
                        buffer.clear();
                    }
                }
            }

            temp_dir.pop();
        }
    }

    crate_dir.push("src");
    temp_dir.push("src");
    fs::create_dir_all(&temp_dir);

    copy_all(
        &crate_dir,
        &mut temp_dir,
        &mut last_opened,
        &mut buffer,
        file,
    );

    crate_dir.pop();
    temp_dir.pop();

    if let Some((mut r, mut f)) = last_opened {
        r.read_to_string(&mut buffer);

        cr = if let Some(i) = buffer.find('\n') {
            if let Some(s) = buffer.get(i - 1..i) {
                s == "\r"
            } else {
                false
            }
        } else {
            false
        };

        start_index = buffer
            .lines()
            .take(start.line - 1)
            .map(|e| e.len() + 1 + (cr as usize))
            .sum::<usize>()
            + start.column;
        let end_index = buffer
            .lines()
            .take(end.line - 1)
            .map(|e| e.len() + 1 + (cr as usize))
            .sum::<usize>()
            + end.column;

        unsafe {
            buffer.as_mut_vec().splice(
                start_index..end_index,
                RESERVED_PATH.as_bytes().iter().copied(),
            );
        }

        /*
                    start_index = buffer[..start_index].rfind('(').unwrap_or(0);
                    start_index = buffer[..start_index].rfind('!').unwrap_or(0);
                    start_index = buffer[..start_index].rfind(|c| !(c.is_alphabetic() && c.is_numeric())).unwrap_or(0);
        */

        let mut close_brace_count = 0_i64;
        let mut close_par_count = 0_i64;
        start_index = buffer[..start_index]
            .rfind(|c| {
                let close_brace = c == '}';
                let close_par = c == ')';
                let open_par = c == '(';
                let open_brace = c == '{';

                close_brace_count += close_brace as i64;
                close_par_count += close_par as i64;
                close_brace_count -= open_brace as i64;
                close_par_count -= open_par as i64;

                open_brace && close_brace_count <= 0 && close_par_count <= 0
            })
            .unwrap_or(0);

        start_index += 1;

        buffer.insert_str(start_index, &import);

        f.write_all(buffer.as_bytes());
    }

    temp_dir.push("Cargo.toml");
    crate_dir.push("Cargo.toml");
    fs::copy(&crate_dir, &temp_dir);
    crate_dir.pop();
    temp_dir.pop();

    drop(crate_dir);
    drop(ctd);

    /*
        temp_dir.pop();
        temp_dir.push("target");

        copy_recursive("target", &temp_dir);

        if let Some(e) = env::var_os("CARGO_HOME") {
            copy_recursive(e, &temp_dir);
        }
    */

    let mut command = Command::new("cargo");

    temp_dir.push(".cargo");
    command.env("CARGO_HOME", temp_dir.as_os_str());
    command.arg("check");
    temp_dir.pop();

    temp_dir.pop();
    drop(temp_dir);
    drop(t);

    if cfg!(not(debug_assertions)) {
        command.arg("--release");
    }

    let stderr = String::from_utf8(command.output().expect("failed to launch program.").stderr)
        .expect("stderr non-utf8.");

    let mut line = 0;
    let mut column = 0;
    let mut index = 0;

    for e in buffer.lines() {
        line += 1;
        index += e.len();

        if index >= start_index {
            column = index - start_index;
            break;
        }

        index += 1 + (cr as usize);
    }

    for c in buffer[start_index..start_index + import.len()].chars() {
        if c == '\r' {
            continue;
        }

        if c == '\n' {
            column = 0;
            line += 1;
            continue;
        }

        if stderr.contains(&format!("{}:{}", line, column)) {
            return false;
        }

        column += 1;
    }

    true
}

use syn::parse_macro_input;

#[proc_macro_attribute]
pub fn if_def(attr: TokenStream, item: TokenStream) -> TokenStream {
    if if_def_internal(parse_macro_input!(attr as syn::Path)) {
        item
    } else {
        TokenStream::new()
    }
}

use proc_macro::quote;

#[proc_macro]
pub fn defined(input: TokenStream) -> TokenStream {
    if if_def_internal(parse_macro_input!(input as syn::Path)) {
        quote!(true)
    } else {
        quote!(false)
    }
}

const CFG_TRUE: &'static str = if cfg!(windows) { "windows" } else { "unix" };

const CFG_FALSE: &'static str = if cfg!(windows) { "unix" } else { "windows" };

#[proc_macro]
pub fn cfg_defined(input: TokenStream) -> TokenStream {
    if if_def_internal(parse_macro_input!(input as syn::Path)) {
        CFG_TRUE.parse().unwrap()
    } else {
        CFG_FALSE.parse().unwrap()
    }
}

/*
fn first_span(x: TokenStream) -> Span {
    use proc_macro::TokenTree::*;

    for e in x {
        return match e {
            Group(x) => x.span(),
            Ident(x) => x.span(),
            Punct(x) => x.span(),
            Literal(x) => x.span(),
        }
    }

    panic!()
}

fn replace_crate_comp(x: &mut syn::Path, crate_rep: syn::Ident) {
    if let Some(e) = x.segments.first() {
        if e.ident == "crate" {
            x.segments.insert(1, crate_rep);
        }
    }
}

fn change_attr(attrs: &mut [syn::Attribute], crate_rep: syn::Ident, features: &[String]) {
    for e in attrs.iter_mut() {
        replace_crate_comp(&mut e.path, crate_rep)
    }
}

fn manage_items(item: &mut syn::Item, crate_rep: syn::Ident, features: &[String]) {

            use syn::Item::*;
                match item {
    Const(ref mut x) => {
        change_attr(&mut x.attrs, crate_rep, features);
    }
    Enum(ref mut x) => {
        change_attr(&mut x.attrs, crate_rep, features);

    }
    ExternCrate(ref mut x) => {
        change_attr(&mut x.attrs, crate_rep, features);

    }
    Fn(ref mut x) => {
        change_attr(&mut x.attrs, crate_rep, features);

    }
    ForeignMod(ref mut x) => {
        change_attr(&mut x.attrs, crate_rep, features);

    }
    Impl(ref mut x) => {
        change_attr(&mut x.attrs, crate_rep, features);

    }
    Macro(ref mut x) => {
        change_attr(&mut x.attrs, crate_rep, features);

    }
    Macro2(ref mut x) => {
        change_attr(&mut x.attrs, crate_rep, features);

    }
    Mod(ref mut x) => {
        change_attr(&mut x.attrs, crate_rep, features);

    }
    Static(ref mut x) => {
        change_attr(&mut x.attrs, crate_rep, features);

    }
    Struct(ref mut x) => {
        change_attr(&mut x.attrs, crate_rep, features);

    }
    Trait(ref mut x) => {
        change_attr(&mut x.attrs, crate_rep, features);

    }
    TraitAlias(ref mut x) => {
        change_attr(&mut x.attrs, crate_rep, features);

    }
    Type(ref mut x) => {
        change_attr(&mut x.attrs, crate_rep, features);

    }
    Union(ref mut x) => {
        change_attr(&mut x.attrs, crate_rep, features);

    }
    Use(ref mut x) => {
        change_attr(&mut x.attrs, crate_rep, features);

    }
    _ => ()
}
}

fn manage_pats(item: &mut syn::Pat, crate_rep: syn::Ident, features: &[String]) {
    use syn::Pat::*;

    match item {
    Box(ref mut x) => {
            change_attr(&mut x.attrs, crate_rep, features);
            manage_pats(&mut x.pat, crate_rep, features);
        }
    Ident(ref mut x) => {
            change_attr(&mut x.attrs, crate_rep, features);

            if let Some((_, ref mut pat)) = *x.subpat {
                manage_pats(pat, crate_rep, features);
            }
        }
    Lit(ref mut x) => {
            change_attr(&mut x.attrs, crate_rep, features);
            manage_exprs(&mut x.expr);
        }
    Macro(ref mut x) => {
            change_attr(&mut x.attrs, crate_rep, features);

        }
    Or(ref mut x) => {
            change_attr(&mut x.attrs, crate_rep, features);

        }
    Path(ref mut x) => {
            change_attr(&mut x.attrs, crate_rep, features);

        }
    Range(ref mut x) => {
            change_attr(&mut x.attrs, crate_rep, features);

        }
    Reference(ref mut x) => {
            change_attr(&mut x.attrs, crate_rep, features);

        }
    Rest(ref mut x) => {
            change_attr(&mut x.attrs, crate_rep, features);

        }
    Slice(ref mut x) => {
            change_attr(&mut x.attrs, crate_rep, features);

        }
    Struct(ref mut x) => {
            change_attr(&mut x.attrs, crate_rep, features);

        }
    Tuple(ref mut x) => {
            change_attr(&mut x.attrs, crate_rep, features);

        }
    TupleStruct(ref mut x) => {
            change_attr(&mut x.attrs, crate_rep, features);

        }
    Type(ref mut x) => {
            change_attr(&mut x.attrs, crate_rep, features);

        }
    Wild(ref mut x) => {
            change_attr(&mut x.attrs, crate_rep, features);

        }
    _ => ()
}
}

fn manage_stmts(item: &mut syn::Expr, crate_rep: syn::Ident, features: &[String]) {
    use syn::Stmt::*;

    match item {
        Local(ref mut x) => {
            change_attr(&mut x.attrs, crate_rep, features);
            manage_pats(&mut x.pat, crate_rep, features);

            if let Some((_, ref mut expr)) = *x.init {
                manage_exprs(expr, crate_rep, features);
            }
        }
        Item(ref mut x) => {
            change_attr(&mut x.attrs, crate_rep, features);

        }
        Expr(ref mut x) => {
            change_attr(&mut x.attrs, crate_rep, features);

        }
        Semi(ref mut x, _) => {
            change_attr(&mut x.attrs, crate_rep, features);

        }
    }
}
fn manage_exprs(item: &mut syn::Expr, crate_rep: syn::Ident, features: &[String]) {
    use syn::Expr::*;

    match item {
    Array(ref mut x) => {
        change_attr(&mut x.attrs, crate_rep, features);

    }
    Assign(ref mut x) => {
        change_attr(&mut x.attrs, crate_rep, features);

    }
    AssignOp(ref mut x) => {
        change_attr(&mut x.attrs, crate_rep, features);

    }
    Async(ref mut x) => {
        change_attr(&mut x.attrs, crate_rep, features);

    }
    Await(ref mut x) => {
        change_attr(&mut x.attrs, crate_rep, features);

    }
    Binary(ref mut x) => {
        change_attr(&mut x.attrs, crate_rep, features);

    }
    Block(ref mut x) => {
        change_attr(&mut x.attrs, crate_rep, features);

    }
    Box(ref mut x) => {
        change_attr(&mut x.attrs, crate_rep, features);

    }
    Break(ref mut x) => {
        change_attr(&mut x.attrs, crate_rep, features);

    }
    Call(ref mut x) => {
        change_attr(&mut x.attrs, crate_rep, features);

    }
    Cast(ref mut x) => {
        change_attr(&mut x.attrs, crate_rep, features);

    }
    Closure(ref mut x) => {
        change_attr(&mut x.attrs, crate_rep, features);

    }
    Continue(ref mut x) => {
        change_attr(&mut x.attrs, crate_rep, features);

    }
    Field(ref mut x) => {
        change_attr(&mut x.attrs, crate_rep, features);

    }
    ForLoop(ref mut x) => {
        change_attr(&mut x.attrs, crate_rep, features);

    }
    Group(ref mut x) => {
        change_attr(&mut x.attrs, crate_rep, features);

    }
    If(ref mut x) => {
        change_attr(&mut x.attrs, crate_rep, features);

    }
    Index(ref mut x) => {
        change_attr(&mut x.attrs, crate_rep, features);

    }
    Let(ref mut x) => {
        change_attr(&mut x.attrs, crate_rep, features);

    }
    Lit(ref mut x) => {
        change_attr(&mut x.attrs, crate_rep, features);

    }
    Loop(ref mut x) => {
        change_attr(&mut x.attrs, crate_rep, features);

    }
    Macro(ref mut x) => {
        change_attr(&mut x.attrs, crate_rep, features);

    }
    Match(ref mut x) => {
        change_attr(&mut x.attrs, crate_rep, features);

    }
    MethodCall(ref mut x) => {
        change_attr(&mut x.attrs, crate_rep, features);

    }
    Paren(ref mut x) => {
        change_attr(&mut x.attrs, crate_rep, features);

    }
    Path(ref mut x) => {
        change_attr(&mut x.attrs, crate_rep, features);

    }
    Range(ref mut x) => {
        change_attr(&mut x.attrs, crate_rep, features);

    }
    Reference(ref mut x) => {
        change_attr(&mut x.attrs, crate_rep, features);

    }
    Repeat(ref mut x) => {
        change_attr(&mut x.attrs, crate_rep, features);

    }
    Return(ref mut x) => {
        change_attr(&mut x.attrs, crate_rep, features);

    }
    Struct(ref mut x) => {
        change_attr(&mut x.attrs, crate_rep, features);

    }
    Try(ref mut x) => {
        change_attr(&mut x.attrs, crate_rep, features);

    }
    TryBlock(ref mut x) => {
        change_attr(&mut x.attrs, crate_rep, features);

    },
    Tuple(ref mut x) => {
        change_attr(&mut x.attrs, crate_rep, features);

    },
    Type(ref mut x) => {
        change_attr(&mut x.attrs, crate_rep, features);

    },
    Unary(ref mut x) => {
        change_attr(&mut x.attrs, crate_rep, features);

    }
    Unsafe(ref mut x) => {
        change_attr(&mut x.attrs, crate_rep, features);

    }
    While(ref mut x) => {
        change_attr(&mut x.attrs, crate_rep, features);

    }
    Yield(ref mut x) => {
        change_attr(&mut x.attrs, crate_rep, features);

    }
    _ => ()
}

fn manage_types(item: &mut syn::Type, crate_rep: syn::Ident, features: &[String]) {

            use syn::Type::*;
                match item {
    Array(ref mut x) {
        manage_types(&mut x.elem, crate_rep, features);
    }
    BareFn(ref mut x) {

    }
    Group(ref mut x) {

    }
    ImplTrait(ref mut x) {

    }
    Infer(ref mut x) {

    }
    Macro(ref mut x) {

    }
    Never(ref mut x) {

    }
    Paren(ref mut x) {

    }
    Path(ref mut x) {

    }
    Ptr(ref mut x) {

    }
    Reference(ref mut x) {

    }
    Slice(ref mut x) {

    }
    TraitObject(ref mut x) {

    }
    Tuple(ref mut x) {

    }
    _ => ()
}
}

static CRATE_NAME: Mutex<String> = Mutex::new(String::new());

fn crate_name() -> String {
    let mut met = cargo_metadata::MetadataCommand::new();

    met.no_deps();
    met.exec().unwrap().packages.swap_remove(0).name
}
lazy_static::lazy_static! {
    static ref CODE_HELPER: Regex = Regex::new(r"(\b)crate(\b)").unwrap();
}

fn get_crates() -> String {
    use std::fmt::Write;
    use cargo_metadata::*;

    // can't panic,if we reached here then the manifest was already checked
    let a = MetadataCommand::new().exec().unwrap();

    temp_dir().or_else(|| env::var("OUT_DIR").ok().map(PathBuf::from)).map(env::set_current_dir);

    let packages = a.packages;
    let mut code = String::new();

    let mut crates_written = Vec::new();

    while let Some(i) = packages.iter().position(|x| {
        for e in x.dependencies.iter() {
            if !crates_written.iter().any(|(name, version)| e.name == name && e.version == version) {
                return false
            }
        }

        true
    }) {
        let Package { targets, version, name, .. } = packages.swap_remove(i);

        crates_written.push((name, version));

        for target in targets {
            let Target { src_path, required_features, kind, name, .. } = target;

            let mut command = Command::new("rustc");
            let mut crate_code = fs::read_to_string(src_path).unwrap();

            if kind.contains("bin") {
                name.push_str("_bin");
            }

            if name == "if_def" {
                command.args(&["--cfg", r#"feature="useless_if_def""#])
            }

            write!(code, "mod {} {{ {} }}", name, CODE_HELPER.replace(&crate_code, |x| {
                format!("{}crate::{}{}", x[1], name, x[2])
            }));

            fs::remove_file("crate");
            let mut f = File::create("crate").unwrap();
            f.write_all(code.as_bytes());
            f.sync_all();

            command.arg("crate");

            required_features.into_iter().for_each(|mut e| {
                unsafe { e.as_mut_vec().splice(0..0, br#"feature=""#) }

                e.push('"');

                command.arg("--cfg");
                command.arg(e);
            });

            command.args(&["-Zunstable-options", "--pretty=expanded"]);

            let expanded = command.output().unwrap();

            code.clear();
            unsafe { code.as_mut_vec().extend(expanded.stdout.iter().copied()) }
        }
    }

    code
}


fn source_file(x: TokenStream) -> PathBuf {
    use proc_macro::TokenTree::*;

    let mut x = x.into_iter();
    match x.next().unwrap() {
        Group(x) => x.span().source_file().path(),
        Ident(x) => x.span().source_file().path(),
        Punct(x) => x.span().source_file().path(),
        Literal(x) => x.span().source_file().path(),
    }
}
*/
