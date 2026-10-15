"""Supported cuDSS configuration experiments; no vendor graph modifications."""

import ctypes as ct
import gc
import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
import warp as wp

from benchmarks.upstream.adapters import matrix
from warpack import CuDSSInverse

from .cudss_scalar_probe import copies
from .nullspace_compare import audit

# Public cuDSS 0.8 enums; this instrumentation is outside production numerics.
PARAMS = dict(
    solve_algorithm=2,
    ir_steps=5,
    hybrid_memory=11,
    hybrid_execute=15,
    superpanels=20,
    deterministic=24,
)
PROFILES = dict(
    default={},
    explicit_zero_ir={"ir_steps": 0},
    general_solve={"solve_algorithm": 1},
    deterministic={"deterministic": 1},
    no_superpanels={"superpanels": 0},
)


@contextmanager
def configuration(profile):
    from warp_cudss import CudssSolver

    def construct(*args, **kwargs):
        solver = CudssSolver(*args, **kwargs)
        for key, value in PROFILES[profile].items():
            data = ct.c_int(value)
            status = solver._lib.cudssConfigSet(
                solver._config, PARAMS[key], ct.byref(data), ct.sizeof(data)
            )
            if status:
                raise RuntimeError(f"Config {key} returned {status}")
        return solver

    with patch("warp_cudss.CudssSolver", construct):
        yield


def get_int(solver, parameter):
    function = solver._lib.cudssDataGet
    function.argtypes = [
        ct.c_void_p,
        ct.c_void_p,
        ct.c_int,
        ct.c_void_p,
        ct.c_size_t,
        ct.POINTER(ct.c_size_t),
    ]
    value, written = ct.c_int(), ct.c_size_t()
    status = function(
        solver._handle,
        solver._data,
        parameter,
        ct.byref(value),
        ct.sizeof(value),
        ct.byref(written),
    )
    return dict(status=status, value=value.value, bytes=written.value)


def config_values(solver):
    function = solver._lib.cudssConfigGet
    function.argtypes = [ct.c_void_p, ct.c_int, ct.c_void_p, ct.c_size_t, ct.POINTER(ct.c_size_t)]
    result = {}
    for name, parameter in PARAMS.items():
        value, written = ct.c_int(), ct.c_size_t()
        status = function(
            solver._config, parameter, ct.byref(value), ct.sizeof(value), ct.byref(written)
        )
        result[name] = dict(status=status, value=value.value, bytes=written.value)
    return result


def run(structure, n, width, profile):
    a = sp.diags(np.arange(2.0, n + 2), format="csr")
    if structure == "chain":
        a += sp.diags([np.full(n - 1, -0.1)] * 2, [-1, 1], shape=(n, n), format="csr")
    elif structure == "dense":
        a = sp.csr_matrix(np.eye(n) * n + np.ones((n, n)))
    with configuration(profile):
        inv = CuDSSInverse(matrix(a), width, mtype="spd", refinement_steps=0)
    rhs = wp.ones_like(inv.x)
    inv.apply(rhs, inv.x)
    wp.synchronize()
    row = dict(
        structure=structure,
        n=n,
        width=width,
        profile=profile,
        config=config_values(inv.solver),
        superpanels=get_int(inv.solver, 17),
    )
    with wp.ScopedCapture() as cap:
        inv.apply(rhs, inv.x)
    row["copies_before"] = copies(cap.graph)
    row["graph_audit"] = audit(
        cap.graph, Path(f"build/cudss-graph-audit/{structure}-{n}-{width}-{profile}.dot")
    )
    checks = []
    for j in range(3):
        expected = np.random.default_rng(j).normal(size=(width, n))
        values = (a @ expected.T).T.copy()
        rhs.assign(values)
        wp.capture_launch(cap.graph)
        output = inv.x.numpy()
        checks.append(
            dict(
                max_error=float(abs(output - expected).max()),
                relative_residual=float(
                    np.linalg.norm((a @ output.T).T - values) / np.linalg.norm(values)
                ),
            )
        )
    row.update(
        checks=checks,
        copies_after=copies(cap.graph),
        accepted=all(c["max_error"] < 1e-12 and c["relative_residual"] < 1e-12 for c in checks),
    )
    cap.graph = None
    gc.collect()
    inv.close()
    return row


def main():
    Path("build/cudss-graph-audit").mkdir(parents=True, exist_ok=True)
    rows = []
    cases = [("diagonal", n, 1, "default") for n in (4, 16, 64)]
    cases += [(s, 64, w, "default") for s in ("chain", "dense") for w in (1, 8)]
    cases += [("chain", 64, 1, p) for p in PROFILES if p != "default"]
    for args in cases:
        row = run(*args)
        rows.append(row)
        Path("build/cudss-graph-audit/config-probe.json").write_text(
            json.dumps(rows, indent=2) + "\n"
        )
        print(
            args,
            row["accepted"],
            row["superpanels"],
            [(r["int32_pair"]) for r in row["copies_before"]],
            flush=True,
        )

    if not all(r["accepted"] and r["copies_before"] == r["copies_after"] for r in rows):
        raise SystemExit("Configuration accuracy or source-lifetime check failed")


if __name__ == "__main__":
    main()
