import os
import shlex
import shutil
import argparse
import subprocess

from llvm_utils import *
from utils import *

from pathlib import Path
from contextlib import contextmanager
from typing import List, Dict, Tuple, Optional, Callable, Union, Any
from datetime import datetime

def emit_cmake_preset(llvm_config: LLVMConfig, vcpkg_toolchain_path: str, cmake_prefix_path: str) -> str:
    arch = str(llvm_config.cmake_arch).lower()
    generator = Generator.short_str(llvm_config.generator).lower()
    preset_name = f"vcpkg-{arch}-{generator}"
    preset = {
        "name": preset_name,
        "displayName": f"[vcpkg; {arch}] {generator}",
        "description": "Generated via build script",
        "generator": str(llvm_config.generator),
        "binaryDir": "${sourceDir}/build/${presetName}",
        "toolchainFile": vcpkg_toolchain_path,
        "cacheVariables": {
            "CMAKE_C_COMPILER": "clang-cl.exe",
            "CMAKE_CXX_COMPILER": "clang-cl.exe",
            "CLOBBER_USE_CRT": "OFF",
            "CLOBBER_STRICT": "OFF",
            "CMAKE_BUILD_TYPE": "Debug",
            "CMAKE_PREFIX_PATH": cmake_prefix_path,
        },
        "environment": {
            "VCPKG_TARGET_TRIPLET": "arm64-windows"
        }
    }
    
    cmake_preset_json = {
        "version": 8,
        "configurePresets": [ preset ]
    }
    
    cmake_preset_json_path = norm(os.path.join(os.getcwd(), "CMakePresets.json"))
    with open(cmake_preset_json_path, "w") as f:
        json.dump(cmake_preset_json, f, indent=2)
    return cmake_preset_json_path
    
def get_vcpkg_toolchain_path() -> str:
    value_opt = os.environ.get("VCPKG_ROOT")
    if value_opt is None:
        return None
    
    # assumes that there is a `vcpkg` executable via path
    vcpkg_path = Path(shutil.which("vcpkg"))
    return Path(os.path.join(str(vcpkg_path.parent), "scripts/buildsystems/vcpkg.cmake")).as_posix()

def clone_and_build_vcpkg(vcpkg_dir: str) -> bool:
    if not os.path.isdir(vcpkg_dir):
        log_info(f"creating directory \"{vcpkg_dir}\"")
        os.mkdir(vcpkg_dir)
        
    git_path = get_git_executable_path()
    clone_args = [git_path, "clone", "https://github.com/microsoft/vcpkg.git", vcpkg_dir]
    bootstrap_args = ["./bootstrap-vcpkg.sh" if os.name != "nt" else "bootstrap-vcpkg.bat"]
    
    log_info(shlex.join(clone_args))
    with scope_hook():
        subprocess.run(clone_args, check=True, cwd=vcpkg_dir)
        
    log_info(shlex.join(bootstrap_args))
    with scope_hook():
        subprocess.run(bootstrap_args, check=True, cwd=vcpkg_dir)

def bootstrap_vcpkg(cwd: str) -> str:
    vcpkg_dir = os.path.join(cwd, "bootstrap", "vcpkg")
    if not (os.path.isdir(vcpkg_dir) and any(os.scandir(vcpkg_dir))):
        clone_and_build_vcpkg(vcpkg_dir)
    else:
        log_info(f"local copy of vcpkg detected")

    vcpkg_exe_path = os.path.join(vcpkg_dir, "vcpkg.exe")
    vcpkg_toolchain_path = os.path.join(vcpkg_dir, "scripts/buildsystems/vcpkg.cmake")
    return vcpkg_exe_path, vcpkg_toolchain_path

def get_vcpkg_executable_path(cwd: str) -> str:
    vcpkg_path = shutil.which("vcpkg")
    if vcpkg_path:
        log_info(f"vcpkg found: \"{vcpkg_path}\"")
        try:
            with scope_hook():
                subprocess.run([vcpkg_path, "--version"], check=True)
            vcpkg_toolchain_path = get_vcpkg_toolchain_path()
            return vcpkg_path, vcpkg_toolchain_path;
        except subprocess.CalledProcessError:
            log_warning("vcpkg found but failed to run, bootstrapping from source.")
            return bootstrap_vcpkg(cwd)
    else:
        log_info("vcpkg not found in PATH, bootstrapping from source.")
        return bootstrap_vcpkg(cwd)
    
def install_vcpkg_packages(vcpkg_executable: str, cwd: str) -> None:
    install_args = [vcpkg_executable, "install"]
    log_info(shlex.join(install_args))
    with scope_hook():
        subprocess.run(install_args, check=True)
        
def get_default_config() -> LLVMConfig:
    return LLVMConfig(
        generator=Generator.VS,
        cmake_arch=CMakeArchitecture.ARM64,
        targets=[LLVMTarget.AARCH64],
        build_config=BuildConfig.DEBUG
    )
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-local-vcpkg", action="store_true", help="Force bootstrapping and usage of local vcpkg instance")
    parser.add_argument("--force-llvm-bootstrap", action="store_true", help="Force LLVM bootstrap task even though it has been detected.")
    parser.add_argument("-G", "--generator", choices=["ninja", "msvc"])
    parser.add_argument("-c", "--config", choices=["Debug", "Release"])
    args = parser.parse_args()
    
    log_info(f"force_llvm_bootstrap: {args.force_llvm_bootstrap}")
    
    try:
        if os.name != "nt":
            log_warning("This bootstrap script is not tested for non-windows machines.")
            log_warning("If you experience any errors, feel free to open an issue at https://github.com/bryjen/clobber/issues")
        
        cwd = Path(os.getcwd()).as_posix()
        
        
        vcpkg_exe_path, vcpkg_toolchain_path = bootstrap_vcpkg(cwd) if args.force_local_vcpkg else get_vcpkg_executable_path(cwd)
        vcpkg_exe_path, vcpkg_toolchain_path = norm(vcpkg_exe_path), norm(vcpkg_toolchain_path)
        log_info(f"vcpkg executable: {vcpkg_exe_path}")
        log_info(f"vcpkg toolchain path: {vcpkg_toolchain_path}")
        
        install_vcpkg_packages(vcpkg_exe_path, cwd)
        
        llvm_config = get_default_config()
        cmake_prefix_path = bootstrap_llvm(llvm_config, force_bootstrap=args.force_llvm_bootstrap)
        
        cmake_presets_json_path = emit_cmake_preset(llvm_config, vcpkg_toolchain_path, cmake_prefix_path)
        log_info(f"generated cmake presets file at: {cmake_presets_json_path}")
    except Exception as e:
        log_error(e)