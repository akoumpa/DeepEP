# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
_NATIVE_EXPORTS = {
    "Buffer",
    "Config",
    "EventOverlap",
    "HybridEPBuffer",
    "HybridEpConfigInstance",
}
_native_import_error = None

try:
    from .utils import EventOverlap as EventOverlap
    from .buffer import Buffer as Buffer
    from .hybrid_ep_buffer import HybridEPBuffer as HybridEPBuffer

    # noinspection PyUnresolvedReferences
    from deep_ep_cpp import Config as Config
    from hybrid_ep_cpp import HybridEpConfigInstance as HybridEpConfigInstance
except ModuleNotFoundError as error:
    if error.name not in {"deep_ep_cpp", "hybrid_ep_cpp"}:
        raise
    _native_import_error = error
    for export_name in _NATIVE_EXPORTS:
        globals().pop(export_name, None)


def native_extensions_available():
    """Return whether the DeepEP CUDA extension modules are installed."""
    return _native_import_error is None


def __getattr__(name):
    if name in _NATIVE_EXPORTS and _native_import_error is not None:
        raise RuntimeError(
            "DeepEP was installed without CUDA extensions. Install on a host with "
            "CUDA_HOME configured, or force a clean native build with "
            "DEEPEP_BUILD_EXTENSIONS=1 and pip --no-cache-dir."
        ) from _native_import_error
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = sorted(_NATIVE_EXPORTS | {"native_extensions_available"})
