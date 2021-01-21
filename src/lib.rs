#![feature(proc_macro_span, thread_spawn_unchecked, proc_macro_quote)]

use ::{
    proc_macro::{Span, TokenStream},
    std::{
        collections::HashMap,
        env,
        fs::{self, File},
        io::{prelude::*, ErrorKind, SeekFrom},
        path::{Path, PathBuf},
        process::Command,
    },
    sync_2::{Mutex, RwLock},
};

static TEMP_DIR: Mutex<Option<PathBuf>> = Mutex::new(None);
static CRATE_DIR: Mutex<Option<PathBuf>> = Mutex::new(None);
static STRATEGY: RwLock<bool> = RwLock::new(false);
static CODE_TABLE: Mutex<Option<HashMap<PathBuf, (String, File)>>> = Mutex::new(None);

fn first_span(x: TokenStream) -> Option<Span> {
    use proc_macro::TokenTree::*;

    x.into_iter().next().map(|e| match e {
        Group(x) => x.span(),
        Ident(x) => x.span(),
        Punct(x) => x.span(),
        Literal(x) => x.span(),
    })
}


#[proc_macro]
pub fn strategy_false(x: TokenStream) -> TokenStream {
    *STRATEGY.write().unwrap() = false;
    x
}

#[proc_macro]
pub fn strategy_true(x: TokenStream) -> TokenStream {
    *STRATEGY.write().unwrap() = true;
    x
}

#[proc_macro]
pub fn strategy_toggle(x: TokenStream) -> TokenStream {
    let mut l = STRATEGY.write().unwrap();
    *l = !*l;
    x
}

fn if_def_internal(input2: TokenStream) -> bool {
    let input = input2.to_string();

    let nested = || env::var("RUST_IF_DEF").is_ok();
    let strategy = STRATEGY.read().unwrap();

    if input == quote!(true).to_string() || if *strategy { nested() } else { false } {
        return true
    }

    if input == quote!(false).to_string() || if *strategy { false } else { nested() } {
        return false
    }

    let span = if let Some(e) = first_span(input2) {
        e
    } else {
        return false;
    };

    let import = format!("#[allow(unused_imports)]use {} as _;", input);

    let mut ctd = CRATE_DIR.lock().unwrap();
    let crate_dir = ctd.get_or_insert_with(|| {
        env::var_os("CARGO_MANIFEST_DIR")
            .map(PathBuf::from)
            .unwrap_or_default()
    });

    let mut t = TEMP_DIR.lock().unwrap();
    let temp_dir = t.get_or_insert_with(|| env::var_os("TMP")
.or_else(|| env::var_os("OUT_DIR")).map(PathBuf::from).or_else(|| Some(PathBuf::default())).map(|mut p| {
            p.push("rust_if_def");
            p
        }).unwrap());

    let start = span.start();
    let end = span.end();
    let p = fs::canonicalize(&span.source_file().path()).unwrap();

    let file = p
        .file_name()
        .expect("maybe not that real file lacks filename");
    
    /*
        let random = || {
            use std::time::*;

            SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_else(|e| e.duration()).as_nanos()
        };

        let mut rand_int = random();
        let mut crate_n = t.to_string();

        loop {
            temp_dir.push(&crate_n);
            if Path::exists(&temp_dir) {
                rand_int = random();
                n.clear();
                write!(n, "{}", rand_int).unwrap();
                temp_dir.pop();
            } else {
                break
            }
        }
    */

    temp_dir.push("crate_n");

    fn copy_all<T: AsRef<Path>>(
        src: T,
        temp_dir: &mut PathBuf,
        file: &Path,
        file_map: &mut HashMap<PathBuf, (String, File)>
    ) {
        for e in fs::read_dir(src).expect("failed to read src") {
            let entry = e.expect("failed to get entry of src");
            let path = entry.path();

            let file_name = path.file_name().unwrap();

            temp_dir.push(file_name);

            if entry.metadata().unwrap().is_dir() {
                match fs::create_dir(&temp_dir) {
                    Ok(()) => (),
                    x => x.expect("failed creating source directory in temp crate"),
                }
                copy_all(&*path, &mut *temp_dir, file.as_ref(), &mut *file_map);
            } else {
                let b = path != file;
                let (ref mut code, ref mut r) = *file_map.entry(path).or_insert_with_key(|e| (fs::read_to_string(e).expect("failed to copy from src,reading a file"), File::create(&temp_dir).expect("failed to copy from src,creating a file")));

                if b {
                    r.write_all(code.as_bytes()).expect("failed to write source code to temp crate");
                    r.set_len(code.len() as _).unwrap();
                    r.seek(SeekFrom::Start(0)).expect("error seeking");
                }
            }

            temp_dir.pop();
        }
    }

    crate_dir.push("src");
    temp_dir.push("src");

    let mut file_map_l = CODE_TABLE.lock().unwrap();
    let file_map = file_map_l.get_or_insert_with(HashMap::new);

    match fs::create_dir_all(&temp_dir) {
        Ok(()) => {
            copy_all(
                    &crate_dir,
                    &mut *temp_dir,
                    &p,
                    &mut *file_map,
                )
        },
        Err(e) if e.kind() == ErrorKind::AlreadyExists => (),
        x => x.expect("failed creating source directory in temp crate"),
    }

    crate_dir.pop();
    temp_dir.pop();

    let (ref mut buffer, ref mut temp_file) = *file_map.get_mut(&p).unwrap();

    let cr = buffer.find('\n').and_then(|i| buffer.get(i - 1..i)).map(|e| e == "\r").unwrap_or(false);

        let mut start_index = buffer
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

        let splice_start = start_index;
        let splice_end = end_index;

        let mut close_brace_count = 0_usize;
        let mut close_par_count = 0_usize;
        start_index = buffer[..start_index]
            .rfind(|c| {
                let close_brace = c == '}';
                let close_par = c == ')';
                let open_par = c == '(';
                let open_brace = c == '{';

                close_brace_count += close_brace as usize;
                close_par_count += close_par as usize;
                close_brace_count = close_brace_count.saturating_sub(open_brace as _);
                close_par_count = close_par_count.saturating_sub(open_par as _);

                open_brace && close_brace_count == 0 && close_par_count == 0
            })
            .unwrap_or(0);

        start_index += 1; 

        buffer.insert_str(start_index, &import);

        temp_file.write_all(buffer.as_bytes())
            .expect("failed to write source code to temp crate");

    temp_dir.push("Cargo.toml");
    crate_dir.push("Cargo.toml");
    fs::copy(&crate_dir, &temp_dir).expect("temp crate failed to receive the Cargo.toml");
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

    command.arg("check");
    command.current_dir(&temp_dir);

    temp_dir.pop();
    temp_dir.push(".cargo");

    command.env("CARGO_HOME", temp_dir.as_os_str());
    command.env("RUST_IF_DEF", "");

    if cfg!(not(debug_assertions)) {
        command.arg("--release");
    }

    let stderr = String::from_utf8(command.output().expect("failed to launch program.").stderr)
        .expect("stderr non-utf8");

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

    let mut result = true;

    for c in buffer[start_index..start_index + import.len()].chars() {
        if c == '\r' {
            continue;
        }

        if c == '\n' {
            column = 0;
            line += 1;
            continue;
        }

        if stderr.contains(&format!(
            "{}:{}:{}",
            Path::new(file).display(),
            line,
            column
        )) {
            result = false;
        }

        column += 1;
    }

    unsafe {
        let buffer = buffer.as_mut_vec();

        buffer.splice(splice_start..splice_end, result.to_string().as_bytes().iter().copied());

        temp_file.seek(SeekFrom::Start(0)).expect("error seeking");
        temp_file.write_all(&buffer[..]).expect("failed to write source code to temp crate");
        temp_file.set_len(buffer.len() as _).unwrap();
    }
    result
}

#[proc_macro_attribute]
pub fn if_def(attr: TokenStream, item: TokenStream) -> TokenStream {
    if if_def_internal(attr) {
        item
    } else {
        TokenStream::new()
    }
}

use proc_macro::quote;

#[proc_macro]
pub fn defined(input: TokenStream) -> TokenStream {
    if if_def_internal(input) {
        quote!(true)
    } else {
        quote!(false)
    }
}

#[proc_macro]
pub fn cfg_defined(input: TokenStream) -> TokenStream {
    if if_def_internal(input) {
        // one of this have to be setted or both,as rust does not support to delete cfg variables
        if cfg!(windows) { quote!(windows) } else { quote!(unix) }
    } else if cfg!(rust_if_def_reserved_1) {
        // note: not quoting here as the `compile_error!` that make the compilation of this
        // procedural macro library fail instead
        r##"compile_error!(r#"`rust_if_def_reserved_1` somehow setted as cfg variable,aborting"#)"##
        .parse().unwrap()
    } else {
        quote!(rust_if_def_reserved_1)
    }
}

/*

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
