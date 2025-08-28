# mm_backend.py
from contextlib import contextmanager

# Allowed backends
PY_NAIVE = "py_naive"   # your existing triple-loop (or current Python impl)
PY_FAST  = "py_fast"    # if you have a block/optimized Python variant
C_BACKEND = "c"         # ctypes C implementation

# Global (module-level) setting
_CURRENT_BACKEND = PY_NAIVE

def get_backend() -> str:
    return _CURRENT_BACKEND

def set_backend(name: str):
    global _CURRENT_BACKEND
    if name not in {PY_NAIVE, PY_FAST, C_BACKEND}:
        raise ValueError(f"Unknown backend '{name}'")
    _CURRENT_BACKEND = name

@contextmanager
def use_backend(name: str):
    prev = get_backend()
    set_backend(name)
    try:
        yield
    finally:
        set_backend(prev)
