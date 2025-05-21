import os
import json
import shlex
import shutil
import subprocess

from pathlib import Path
from contextlib import contextmanager
from typing import List, Dict, Tuple, Optional, Callable, Union, Any
from datetime import datetime

ansi_grey = "\033[90m";
ansi_yellow = "\033[93m";
ansi_red = "\033[91m";
ansi_reset = "\033[0m";

@contextmanager
def scope_hook():
    """ Change stdout color, differentiates script output from subprocess output """
    print(ansi_grey)
    try:
        yield
    finally:
        print(ansi_reset)
        
def norm(path: str) -> str:
    return Path(path).as_posix()
        
def log_info(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [INF] {message}")
    
def log_warning(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{ansi_yellow}WRN{ansi_reset}] {message}")
    
def log_error(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{ansi_red}ERR{ansi_reset}] {message}")
    
def is_executable(file_path):
    if os.name == "nt":  # windows
        return file_path.lower().endswith((".exe", ".bat", ".cmd"))
    else:  # unix
        return os.path.isfile(file_path) and os.access(file_path, os.X_OK)
    
    
def get_git_executable_path() -> str:
    git_path = shutil.which("git")
    if not git_path:
        raise Exception("Could not find a `git` executable.")
    return git_path

def get_cmake_executable_path() -> str:
    cmake_path = shutil.which("cmake")
    if not cmake_path:
        raise Exception("Could not find a `cmake` executable.")
    return cmake_path