"""
Utility module for bootstrapping and linking LLVM and subprojects.

Some vcpkg triplets (e.g., WOA64) may fail. 
Additionally, due to the size functionality supporting linking to prebuilt LLVM to avoid full rebuilds would be nice.
"""

import os
import json
import shlex
import shutil
import subprocess

from utils import *

from enum import Enum
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable, Union, Any

class Generator(Enum):
    NINJA = 0
    VS = 1
    
    def __str__(self):
        return {
            Generator.NINJA: "Ninja",
            Generator.VS: "Visual Studio 17 2022",
        }[self]
        
    @staticmethod
    def short_str(generator):
        return {
            Generator.NINJA: "ninja",
            Generator.VS: "msvc",
        }[generator]
        
class CMakeArchitecture(Enum):
    X64 = 0
    WIN32 = 1
    ARM64 = 2
    
    def __str__(self):
        return {
            CMakeArchitecture.X64: "x64",
            CMakeArchitecture.WIN32: "Win32",
            CMakeArchitecture.ARM64: "ARM64",
        }[self]
        
class LLVMTarget(Enum):
    X86 = 0
    AARCH64 = 1
    ARM = 2
    
    def __str__(self):
        return {
            LLVMTarget.X86: "X86",
            LLVMTarget.AARCH64: "AArch64",
            LLVMTarget.ARM: "ARM",
        }[self]
        
class BuildConfig(Enum):
    DEBUG = 0
    RELEASE = 1
    RELWITHDEBINFO = 2
    
    def __str__(self):
        return {
            BuildConfig.DEBUG: "Debug",
            BuildConfig.RELEASE: "Release",
            BuildConfig.RELWITHDEBINFO: "RelWithDebInfo",
        }[self]
        
@dataclass
class LLVMConfig:
    generator: Generator
    cmake_arch: CMakeArchitecture 
    targets: List[LLVMTarget]
    build_config: BuildConfig
    

def __get_llvm_dir() -> str:
    return Path(os.getcwd(), "bootstrap/llvm").as_posix()

def __get_llvm_build_dir() -> str:
    return Path(os.getcwd(), "bootstrap/llvm/build").as_posix()


def __llvm_exists() -> bool:
    llvm_dir = __get_llvm_dir()
    return os.path.isdir(llvm_dir) and any(os.scandir(llvm_dir))

def __clone_llvm() -> None:
    llvm_dir = __get_llvm_dir()
    if not os.path.isdir(llvm_dir):
        log_info(f"creating directory \"{llvm_dir}\"")
        os.mkdir(llvm_dir)
        
    git_path = get_git_executable_path()
    clone_args = [git_path, "clone", "https://github.com/llvm/llvm-project.git", llvm_dir]
    
    log_info("cloning llvm monorepo")
    log_info(shlex.join(clone_args))
    with scope_hook():
        subprocess.run(clone_args, check=True, cwd=llvm_dir)
        
        
def __llvm_configured() -> bool:
    llvm_build_dir = __get_llvm_build_dir()
    return os.path.isdir(llvm_build_dir) and os.path.isfile(os.path.join(__get_llvm_build_dir(), "CMakeCache.txt"))

def __configure_llvm(llvm_config: LLVMConfig) -> None:
    llvm_build_dir = __get_llvm_build_dir()
    if not os.path.isdir(llvm_build_dir):
        log_info(f"creating build directory \"{llvm_build_dir}\"")
        os.mkdir(llvm_build_dir)
        
    cmake_path = get_cmake_executable_path()
    llvm_targets_str = ";".join(str(llvm_target) for llvm_target in llvm_config.targets)
        
    llvm_configure_args = [
        cmake_path, "../llvm",
        "-G", str(llvm_config.generator),
        "-A", str(llvm_config.cmake_arch),
        # "-DCMAKE_BUILD_TYPE=Debug",  # not needed in MSVC
        "-DLLVM_ENABLE_PROJECTS=mlir",
        f"-DLLVM_TARGETS_TO_BUILD={llvm_targets_str}",
        "-DLLVM_BUILD_TOOLS=OFF",
        "-DLLVM_INCLUDE_UTILS=OFF",
        "-DLLVM_ENABLE_RTTI=ON",
        "-DLLVM_BUILD_LLVM_DYLIB=ON",
        "-DMLIR_ENABLE_EXECUTION_ENGINE=ON",
        "-DLLVM_INCLUDE_TESTS=OFF",
        "-DLLVM_INCLUDE_DOCS=OFF",
        "-DLLVM_INCLUDE_BENCHMARKS=OFF",
        "-DLLVM_INCLUDE_EXAMPLES=OFF",
        "-DMLIR_INCLUDE_EXAMPLES=OFF",
        "-DLLVM_ENABLE_ASSERTIONS=ON",
        "-DLLVM_ENABLE_TERMINFO=OFF",
        "-DLLVM_ENABLE_ZLIB=OFF",
        "-DMLIR_ENABLE_BINDINGS_PYTHON=OFF",
        "-DCMAKE_INSTALL_PREFIX=../mlir-install"
    ]
    
    log_info("configuring llvm monorepo")
    log_info(shlex.join(llvm_configure_args))
    with scope_hook():
        subprocess.run(llvm_configure_args, check=True, cwd=llvm_build_dir)


def __llvm_built(llvm_config: LLVMConfig) -> bool:
    llvm_build_dir = __get_llvm_build_dir()
    
    if llvm_config.generator == Generator.NINJA:
        # check if executables are present in `{LLVM_BUILD_DIR}/bin`
        return any(is_executable(os.path.join(llvm_build_dir, f)) for f in os.listdir(llvm_build_dir))
    elif llvm_config.generator == Generator.VS:
        # check if executables are present in any of `{LLVM_BUILD_DIR}/Debug/bin`, `{LLVM_BUILD_DIR}/Release/bin`, etc.
        dir_names = [ "Debug", "Release" ]
        abs_dirs = [ Path(os.path.join(llvm_build_dir, dir_name)).as_posix() for dir_name in dir_names ]
        
        for dir in abs_dirs:
            for dirpath, _, filenames in os.walk(dir):
                for name in filenames:
                    if is_executable(os.path.join(dirpath, name)):
                        return True
    else:
        return False

def __build_llvm(llvm_config: LLVMConfig) -> bool:
    llvm_build_dir = __get_llvm_build_dir()
    cmake_path = get_cmake_executable_path()
    
    llvm_build_args = [
        cmake_path, 
        "--build", ".", 
        "--config", str(llvm_config.build_config),
        "--target", "install",
        "--parallel"
    ]
    
    log_info("building llvm monorepo")
    log_info(shlex.join(llvm_build_args))
    with scope_hook():
        subprocess.run(llvm_build_args, check=True, cwd=llvm_build_dir)
        
def __get_cmake_prefix_path() -> str:
    llvm_dir = __get_llvm_dir()
    return norm(os.path.join(llvm_dir, "mlir-install"))
        
        
def bootstrap_llvm(llvm_config: LLVMConfig, force_bootstrap: bool = False) -> Tuple[List[str], List[str]]:
    if (not __llvm_exists()):
        __clone_llvm(llvm_config)
    else:
        log_info(f"local llvm copy detected")
    
    if not __llvm_configured() or force_bootstrap:
        __configure_llvm(llvm_config)
    else:
        log_info(f"local llvm copy is already configured")
    
    if not __llvm_built(llvm_config) or force_bootstrap:
        __build_llvm(llvm_config)
    else:
        log_info(f"local llvm copy is already built")
        
    # return __get_include_directories(llvm_config), __get_link_directories(llvm_config)
    return __get_cmake_prefix_path()