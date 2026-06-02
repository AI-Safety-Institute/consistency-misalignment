"""Locate the cuda-compat lib dir for CUDA forward compatibility (gpt-oss)."""

from __future__ import annotations

import os
from pathlib import Path

ENV_VAR = "CONSISTENCY_EM_CUDA_COMPAT_DIR"


def forward_compat_ld_library_path() -> str | None:
    """Return the cuda-compat lib dir to prepend to ``LD_LIBRARY_PATH``, or None.

    gpt-oss needs CUDA forward compatibility — the cuda-compat ``libcuda``
    ahead of the system one — to load its MXFP4 / MoE kernels on a capped-CUDA
    driver. The directory is read from the ``CONSISTENCY_EM_CUDA_COMPAT_DIR``
    env var and returned only when it exists on this host; otherwise None, a
    no-op for hosts (and the dense models) that don't need it. The path is host
    specific and never hardcoded — see the HPC docs.
    """
    compat_dir = os.environ.get(ENV_VAR)
    if compat_dir and Path(compat_dir).is_dir():
        return compat_dir
    return None
