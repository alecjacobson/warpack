"""Fingerprint callable implementations, including forwarded Warp exports."""

import hashlib
import inspect
from pathlib import Path


def source_hash(symbol):
    return hashlib.sha256(Path(inspect.getfile(symbol)).read_bytes()).hexdigest()


def warp_runtime_hashes(spectra_binary):
    import warp.optim.linear as linear
    import warp.sparse as sparse

    binary = Path(spectra_binary)
    return dict(
        warp_linear=source_hash(linear),
        warp_cg_implementation=source_hash(linear.cg),
        warp_preconditioner_implementation=source_hash(linear.preconditioner),
        warp_fsai_implementation=source_hash(linear.FSAI) if hasattr(linear, "FSAI") else None,
        warp_sparse_implementation=source_hash(sparse.bsr_mv),
        spectra_binary=hashlib.sha256(binary.read_bytes()).hexdigest() if binary.exists() else None,
    )


def complete_warp_fingerprint(hashes):
    """Old re-export-only reports are not automatically comparable."""
    return (
        isinstance(hashes, dict)
        and "warp_fsai_implementation" in hashes
        and all(
            hashes.get(key)
            for key in (
                "warp_cg_implementation",
                "warp_preconditioner_implementation",
                "warp_sparse_implementation",
            )
        )
    )
