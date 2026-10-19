"""Compatibility imports for historical block benchmarks; numerics live in core."""

from warpack.block import (
    BlockLOBPCG,
    Identity,
    ProjectedSolver,
    UpstreamPreconditioner,
    gram,
    rotate,
)

__all__ = ["BlockLOBPCG", "Identity", "ProjectedSolver", "UpstreamPreconditioner", "gram", "rotate"]
