"""Ports AnteZero's oracle (sim/oracle/src/lua.rs, Windows LoadLibraryExW) to Linux dlopen.

AnteZero is namecallfilter's (https://github.com/namecallfilter/antezero, MIT). This script is ours
(Claude's, reviewed by chud): it writes sim/oracle/src/lua_linux.rs into a pinned clone and points lib.rs
and oracle.rs at it, so their oracle runs the game's Lua in a LuaJIT 2.0.5 built from source.
Usage: python3 port_oracle_linux.py <antezero clone>
"""
import re, sys, pathlib
root = pathlib.Path(sys.argv[1]) / "sim/oracle/src"
src = (root / "lua.rs").read_text()
src = src.replace("use std::os::windows::ffi::OsStrExt;\n", "use std::os::unix::ffi::OsStrExt;\n")
src = src.replace("use std::{iter, mem, ptr, slice};", "use std::{mem, slice};")
win = src[src.index('#[link(name = "kernel32")]'):src.index("const LOAD_WITH_ALTERED_SEARCH_PATH: u32 = 0x0000_0008;\n") + len("const LOAD_WITH_ALTERED_SEARCH_PATH: u32 = 0x0000_0008;\n")]
src = src.replace(win, '''#[link(name = "dl")]
unsafe extern "C" {
	fn dlopen(name: *const c_char, flags: c_int) -> *mut ModuleHandle;
	fn dlsym(module: *mut ModuleHandle, name: *const c_char) -> *mut c_void;
	fn dlclose(module: *mut ModuleHandle) -> c_int;
	fn dlerror() -> *const c_char;
}
const RTLD_NOW: c_int = 2;
const RTLD_LOCAL: c_int = 0;
''')
src = src.replace("unsafe { FreeLibrary(self.0) };", "unsafe { dlclose(self.0) };")
start = src.index("		// `LOAD_WITH_ALTERED_SEARCH_PATH` is undefined")
end = src.index("		let module = Module(handle);")
src = src[:start] + '''		let dll = path::absolute(dll).map_err(|e| format!("{}: {e}", dll.display()))?;
		let mut bytes = dll.as_os_str().as_bytes().to_vec();
		bytes.push(0);
		// SAFETY: `bytes` is a NUL-terminated path that outlives the call.
		let handle = unsafe { dlopen(bytes.as_ptr().cast::<c_char>(), RTLD_NOW | RTLD_LOCAL) };
		if handle.is_null() {
			// SAFETY: dlerror returns a static NUL-terminated message after a failed dlopen.
			let msg = unsafe { CStr::from_ptr(dlerror()) }.to_string_lossy().into_owned();
			return Err(format!("could not load {}: {msg}", dll.display()));
		}
''' + src[end:]
src = src.replace("GetProcAddress(module.0,", "dlsym(module.0,")
src = "//! Linux port of lua.rs (dlopen in place of LoadLibraryExW), written by balatro-headless's port_oracle_linux.py.\n" + src
(root / "lua_linux.rs").write_text(src)
lib = (root / "lib.rs").read_text()
lib = lib.replace('#[cfg(not(windows))]\n#[path = "lua_unavailable.rs"]\nmod lua;',
                  '#[cfg(not(windows))]\n#[path = "lua_linux.rs"]\nmod lua;')
(root / "lib.rs").write_text(lib)
o = (root / "oracle.rs").read_text()
o = o.replace("cfg!(windows) && default_dll().exists()", "default_dll().exists()")
(root / "oracle.rs").write_text(o)
print("ported")
