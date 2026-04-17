import ctypes
import site
from pathlib import Path
from typing import Iterable, Sequence, Tuple, List, Set


CUDA_RUNTIME_LIB_RELPATHS: Tuple[Tuple[str, ...], ...] = (
    ("cuda_runtime", "lib", "libcudart.so.12"),
    ("cuda_nvrtc", "lib", "libnvrtc.so.12"),
    ("cuda_nvrtc", "lib", "libnvrtc-builtins.so.12.8"),
    ("nvjitlink", "lib", "libnvJitLink.so.12"),
    ("curand", "lib", "libcurand.so.10"),
    ("cublas", "lib", "libcublas.so.12"),
    ("cublas", "lib", "libcublasLt.so.12"),
    ("cudnn", "lib", "libcudnn.so.9"),
    ("cufft", "lib", "libcufft.so.11"),
    ("cusolver", "lib", "libcusolver.so.11"),
    ("cusparse", "lib", "libcusparse.so.12"),
    ("cusparselt", "lib", "libcusparseLt.so.0"),
)

_CUDART_RELPATHS: Tuple[Tuple[str, ...], ...] = (
    ("cuda_runtime", "lib", "libcudart.so.12"),
    ("cuda_runtime", "lib", "libcudart.so"),
)

_CUDART_LIBRARY_NAMES: Tuple[str, ...] = (
    "libcudart.so.12",
    "libcudart.so",
)


def _iter_site_roots() -> Iterable[Path]:
    seen: set[Path] = set()
    roots: list[str] = []
    try:
        roots.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        user_site = site.getusersitepackages()
        if isinstance(user_site, str):
            roots.append(user_site)
        else:
            roots.extend(user_site)
    except Exception:
        pass

    for root in roots:
        path = Path(root)
        if path in seen:
            continue
        seen.add(path)
        yield path


def _load_cdll(path: str) -> ctypes.CDLL:
    mode = getattr(ctypes, "RTLD_GLOBAL", None)
    if mode is None:
        return ctypes.CDLL(path)
    return ctypes.CDLL(path, mode=mode)


def preload_cuda_runtime_libraries(
    lib_relpaths: Sequence[Tuple[str, ...]] = CUDA_RUNTIME_LIB_RELPATHS,
) -> List[str]:
    """Preload packaged CUDA shared libraries from site-packages/nvidia."""
    loaded: List[str] = []
    seen: Set[str] = set()
    for root in _iter_site_roots():
        nvidia_root = root / "nvidia"
        for rel in lib_relpaths:
            lib_path = nvidia_root.joinpath(*rel)
            lib_key = str(lib_path)
            if lib_key in seen or not lib_path.exists():
                continue
            _load_cdll(lib_key)
            seen.add(lib_key)
            loaded.append(lib_key)
    return loaded


def load_cudart_library() -> ctypes.CDLL:
    """Load CUDA runtime from packaged wheels first, then fall back to system lookup."""
    last_error: OSError | None = None

    for root in _iter_site_roots():
        nvidia_root = root / "nvidia"
        for rel in _CUDART_RELPATHS:
            lib_path = nvidia_root.joinpath(*rel)
            if not lib_path.exists():
                continue
            try:
                return _load_cdll(str(lib_path))
            except OSError as exc:
                last_error = exc

    for lib_name in _CUDART_LIBRARY_NAMES:
        try:
            return _load_cdll(lib_name)
        except OSError as exc:
            last_error = exc

    if last_error is None:
        raise OSError("Unable to locate a CUDA runtime library candidate.")
    raise OSError(
        "Unable to load CUDA runtime library. Tried packaged wheel paths under "
        "site-packages/nvidia and system names libcudart.so.12/libcudart.so."
    ) from last_error
