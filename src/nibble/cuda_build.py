"""Keep CUDA build caches in scratch and expose installed cuRAND headers.

FlashInfer sampling compiles lazily. Della's CUDA compiler installation does
not include cuRAND headers, but the existing PyTorch NVIDIA wheel does.
This changes build configuration only, preserving the fork's sampler.
"""

import argparse
import importlib.util
import os
from pathlib import Path
import shlex


def build_environment(root):
    """Return validated compiler/cache settings without touching a GPU."""
    root = Path(root).resolve()
    package = importlib.util.find_spec("nvidia.curand")
    if package is None or not package.submodule_search_locations:
        raise RuntimeError("installed NVIDIA cuRAND header package is missing")
    include = Path(next(iter(package.submodule_search_locations))) / "include"
    for name in ("curand.h", "curand_kernel.h"):
        if not (include / name).is_file():
            raise RuntimeError(f"cuRAND compiler header is missing: {include / name}")
    flags = os.environ.get("FLASHINFER_EXTRA_CUDAFLAGS", "")
    include_flag = f"-I{include}"
    if include_flag not in flags.split():
        flags = f"{flags} {include_flag}".strip()
    cache = root / "work/cache"
    return {
        "FLASHINFER_EXTRA_CUDAFLAGS": flags,
        "FLASHINFER_WORKSPACE_BASE": str(cache / "flashinfer"),
        "TRITON_CACHE_DIR": str(cache / "triton"),
        "TORCH_EXTENSIONS_DIR": str(cache / "torch-extensions"),
        "XDG_CACHE_HOME": str(cache / "xdg"),
        "CUDA_CACHE_PATH": str(cache / "cuda-driver"),
        "MAX_JOBS": os.environ.get("MAX_JOBS", "2"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(args.root.resolve()):
        raise ValueError("compiler environment file must stay under Nibble root")
    output.parent.mkdir(parents=True, exist_ok=True)
    settings = build_environment(args.root)
    output.write_text(
        "".join(
            f"export {key}={shlex.quote(value)}\n" for key, value in settings.items()
        )
    )
    print(output)


if __name__ == "__main__":
    main()
