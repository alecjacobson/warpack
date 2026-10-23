"""Exact procedural block-comparison inputs; hashing stays outside solver timing.

Anvil retains its existing shared-fixture protocol. These identities cover
canonical scalar operators, actual starting columns, and the reference arrays
used by procedural FEM/Hodge validators. They are not installed solver code.
"""

from operator import index

import numpy as np
import scipy.sparse as sp

from benchmarks.anvil_inputs import array_identity, matrix_identity


def initial_pool(n, width, pool_width=None):
    n, width = index(n), index(width)
    pool_width = width if pool_width is None else index(pool_width)
    if n < 1 or width < 1 or pool_width < width:
        raise ValueError("positive dimensions and initial pool width >= solver width required")
    return np.random.default_rng(42).normal(size=(n, pool_width))


def initial_block(n, width, pool_width=None):
    return np.ascontiguousarray(initial_pool(n, width, pool_width)[:, :width])


def reference_identity(problem, guard=None):
    if problem.expected is None:
        raise ValueError("the procedural validator needs explicit reference values")
    return dict(
        values=array_identity(problem.expected),
        requested_basis=None
        if problem.requested_basis is None
        else array_identity(problem.requested_basis),
        null_basis=None if problem.null_basis is None else array_identity(problem.null_basis),
        nullity=int(problem.nullity),
        requested_count=int(problem.k),
        positive_guard_required=bool(problem.require_positive_guard),
        guard=None if guard is None else array_identity(guard),
    )


def device_host_matrix(a):
    """Read a scalar or block BSR fixture once, outside setup/capture/replay timers."""
    a.nnz_sync()
    values = a.values.numpy()[: a.nnz].reshape(a.nnz, *a.block_shape)
    return sp.bsr_matrix(
        (values, a.columns.numpy()[: a.nnz], a.offsets.numpy()), shape=a.shape
    ).tocsr()


def identities(a, mass, shifted, reference, initial, cupy_width=None, pool=None):
    n = a.shape[0]
    if a.shape != (n, n) or shifted.shape != a.shape or initial.shape[0] != n:
        raise ValueError("inconsistent input dimensions")
    if mass is not None and mass.shape != a.shape:
        raise ValueError("inconsistent mass dimensions")
    if cupy_width is not None and not 1 <= cupy_width <= initial.shape[1]:
        raise ValueError("CuPy start must be a nonempty prefix of the Warp start")
    pool = initial if pool is None else pool
    initial_identity = array_identity(initial)
    if pool.ndim != 2 or array_identity(pool[:, : initial.shape[1]]) != initial_identity:
        raise ValueError("Warp start must match the common pool prefix exactly")
    return dict(
        schema=1,
        kind="procedural_block",
        operators=dict(
            operator=matrix_identity(a),
            mass=dict(kind="identity", shape=[n, n]) if mass is None else matrix_identity(mass),
            shifted=matrix_identity(shifted),
        ),
        reference=reference,
        initial=dict(
            pool=array_identity(pool),
            warp=initial_identity,
            cupy=None if cupy_width is None else array_identity(initial[:, :cupy_width]),
        ),
    )


def collect(
    a, mass, shifted, host_a, host_mass, shift, reference, initial, cupy_width=None, pool=None
):
    """Certify actual device operators against the independent host validation pencil."""
    expected_shift = host_a + shift * (
        sp.eye(host_a.shape[0], format="csr") if host_mass is None else host_mass
    )
    expected = identities(host_a, host_mass, expected_shift, reference, initial, cupy_width, pool)
    actual = identities(
        device_host_matrix(a),
        None if mass is None else device_host_matrix(mass),
        device_host_matrix(shifted),
        reference,
        initial,
        cupy_width,
        pool,
    )
    if actual["operators"] != expected["operators"]:
        raise ValueError("device operators differ from the host validation pencil")
    return actual


def signature(record):
    if (
        not isinstance(record, dict)
        or record.get("schema") != 1
        or record.get("kind") != "procedural_block"
        or not isinstance(record.get("operators"), dict)
        or not all(record["operators"].get(k) for k in ("operator", "mass", "shifted"))
        or not isinstance(record.get("reference"), dict)
        or not record["reference"].get("values")
        or not {
            "requested_basis",
            "null_basis",
            "requested_count",
            "nullity",
            "guard",
            "positive_guard_required",
        }.issubset(record["reference"])
        or not isinstance(record.get("initial"), dict)
        or not record["initial"].get("warp")
        or not record["initial"].get("pool")
        or "cupy" not in record["initial"]
    ):
        raise ValueError(
            "Missing verified procedural input identities; use a historical checkout or new baseline"
        )
    return record
