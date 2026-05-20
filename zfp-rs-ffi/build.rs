use std::{
    collections::BTreeMap,
    env,
    fmt::Write,
    fs,
    path::{Path, PathBuf},
};

#[derive(Debug)]
struct FnSig {
    name: String,
    args: Vec<String>,
    ret: String,
}

fn main() {
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set"));
    let bindings = find_zfp_sys_bindings(&out_dir).unwrap_or_else(|| {
        panic!(
            "could not find zfp-sys bindings.rs under {}",
            out_dir.display()
        )
    });
    println!("cargo:rerun-if-changed={}", bindings.display());

    let source = fs::read_to_string(&bindings).expect("read zfp-sys bindings");
    let signatures = parse_bindings(&source);
    let generated = render_assertions(&signatures);
    fs::write(out_dir.join("abi_signature_assertions.rs"), generated)
        .expect("write generated ABI signature assertions");
}

fn find_zfp_sys_bindings(out_dir: &Path) -> Option<PathBuf> {
    for ancestor in out_dir.ancestors() {
        if ancestor.file_name().is_some_and(|name| name == "build") {
            let mut candidates = Vec::new();
            for entry in fs::read_dir(ancestor).ok()? {
                let entry = entry.ok()?;
                if !entry.file_name().to_string_lossy().starts_with("zfp-sys-") {
                    continue;
                }
                let candidate = entry.path().join("out").join("bindings.rs");
                if candidate.exists() {
                    candidates.push(candidate);
                }
            }
            candidates.sort_by_key(|path| {
                fs::metadata(path)
                    .and_then(|metadata| metadata.modified())
                    .ok()
            });
            return candidates.pop();
        }
    }
    None
}

fn parse_bindings(source: &str) -> BTreeMap<String, FnSig> {
    let mut signatures = BTreeMap::new();
    let mut lines = source.lines().peekable();

    while let Some(line) = lines.next() {
        let trimmed = line.trim();
        if !trimmed.starts_with("pub fn ") {
            continue;
        }

        let mut declaration = trimmed.to_owned();
        while !declaration.ends_with(';') {
            let next = lines.next().expect("function declaration terminates");
            declaration.push(' ');
            declaration.push_str(next.trim());
        }

        let name = fn_name(&declaration);
        if name.starts_with("zfp_") || name.starts_with("stream_") {
            let sig = parse_fn_declaration(&declaration);
            signatures.insert(sig.name.clone(), sig);
        }
    }

    signatures
}

fn fn_name(declaration: &str) -> String {
    let declaration = declaration
        .trim()
        .strip_prefix("pub fn ")
        .expect("pub fn declaration");
    let name_end = declaration.find('(').expect("function has argument list");
    declaration[..name_end].to_owned()
}

fn parse_fn_declaration(declaration: &str) -> FnSig {
    let declaration = declaration
        .trim()
        .strip_prefix("pub fn ")
        .expect("pub fn declaration")
        .trim_end_matches(';')
        .trim();
    let name_end = declaration.find('(').expect("function has argument list");
    let name = declaration[..name_end].to_owned();
    let args_end = declaration.rfind(')').expect("function has closing paren");
    let args = declaration[name_end + 1..args_end]
        .split(',')
        .map(str::trim)
        .filter(|arg| !arg.is_empty())
        .map(|arg| {
            let (_, ty) = arg.split_once(':').expect("argument has type");
            canonical_type(ty.trim())
        })
        .collect();
    let ret = declaration[args_end + 1..]
        .trim()
        .strip_prefix("->")
        .map_or_else(|| "()".to_owned(), |s| canonical_type(str::trim(s)));

    FnSig { name, args, ret }
}

fn canonical_type(ty: &str) -> String {
    let ty = ty.trim();
    if let Some(inner) = ty.strip_prefix("*mut ") {
        return format!("*mut {}", canonical_type(inner));
    }
    if let Some(inner) = ty.strip_prefix("*const ") {
        return format!("*const {}", canonical_type(inner));
    }

    match ty {
        "::std::os::raw::c_void" | "bitstream" => "std::ffi::c_void".to_owned(),
        "::std::os::raw::c_int" | "int32" => "i32".to_owned(),
        "::std::os::raw::c_uint" => "u32".to_owned(),
        "uint" => "uint".to_owned(),
        "uint64" => "uint64".to_owned(),
        "int8" => "i8".to_owned(),
        "uint8" => "u8".to_owned(),
        "int16" => "i16".to_owned(),
        "uint16" => "u16".to_owned(),
        "int64" => "i64".to_owned(),
        "bitstream_offset" => "bitstream_offset".to_owned(),
        "bitstream_size" => "bitstream_size".to_owned(),
        "bitstream_count" => "bitstream_count".to_owned(),
        "zfp_bool" => "zfp_bool".to_owned(),
        "zfp_exec_policy" => "zfp_exec_policy".to_owned(),
        "zfp_mode" => "zfp_mode".to_owned(),
        "zfp_type" => "zfp_type".to_owned(),
        "zfp_config" => "zfp_config".to_owned(),
        "zfp_field" => "zfp_field".to_owned(),
        "zfp_stream" => "zfp_stream".to_owned(),
        "f32" | "f64" | "isize" | "usize" | "u8" | "u16" | "u32" | "u64" | "i8" | "i16" | "i32"
        | "i64" | "()" => ty.to_owned(),
        _ => panic!("unmapped zfp-sys type `{ty}`"),
    }
}

fn render_assertions(signatures: &BTreeMap<String, FnSig>) -> String {
    let mut out = String::from(
        "// @generated by zfp-rs-ffi/build.rs from zfp-sys bindings.rs\n\
         use zfp_rs_ffi::*;\n\n",
    );

    for sig in signatures.values() {
        let args = sig.args.join(", ");
        write!(
            out,
            "#[allow(dead_code, clippy::type_complexity)]\n\
             const _: unsafe extern \"C\" fn({args}) -> {} = {};\n",
            sig.ret, sig.name
        )
        .unwrap();
    }

    out
}
