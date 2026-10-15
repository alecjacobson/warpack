"""Issue 46 probe: shifted metric QR on the same cuDSS block inverse."""

import argparse
import json
import types
from pathlib import Path

import numpy as np
import warp as wp

from warpack import kernels as K

from .compare import timed
from .hodge import make_hodge_case, weak_component_reference
from .runner import warp_setup


@wp.kernel
def shift_gram(gram: wp.array2d[wp.float64], relative: wp.float64):
    trace = wp.float64(0.0)
    for i in range(gram.shape[0]):
        trace += gram[i, i]
    for i in range(gram.shape[0]):
        gram[i, i] += relative * trace


def shifted_orthogonalize(self):
    for pass_index in range(self.shifted_passes + 2):
        self.mass.apply(self.q, self.bq)
        self.gram(self.q, self.bq)
        if pass_index < self.shifted_passes:
            wp.launch(shift_gram, 1, [self.g, self.relative_shift], device=self.device)
        wp.launch(K.cholesky, 1, [self.g, self.l, self.status], device=self.device)
        wp.launch(K.triangular, self.n, [self.q, self.l], device=self.device)


def legacy_orthogonalize(self):
    for _ in range(2):
        self.mass.apply(self.q, self.bq)
        self.gram(self.q, self.bq)
        wp.launch(K.cholesky, 1, [self.g, self.l, self.status], device=self.device)
        wp.launch(K.triangular, self.n, [self.q, self.l], device=self.device)


def run(scale, shift, shifted_passes):
    case = make_hodge_case("weak_positive", weak_scale=scale)
    p = case.problem
    expected, space = weak_component_reference(case, scale)
    solver, inverse = warp_setup(p)
    if shift:
        solver.relative_shift, solver.shifted_passes = shift, shifted_passes
        solver.orthogonalize = types.MethodType(shifted_orthogonalize, solver)
    else:
        solver.orthogonalize = types.MethodType(legacy_orthogonalize, solver)
    graph = solver.capture(100)
    seconds, _ = timed(lambda: wp.capture_launch(graph))
    values, vectors = solver.eigenvalues.numpy(), solver.eigenvectors.numpy().T
    finite = bool(np.isfinite(values).all() and np.isfinite(vectors).all())
    row = dict(
        weak_scale=scale,
        relative_gram_shift=shift,
        shifted_passes=shifted_passes,
        n=p.n,
        k=p.k,
        ncv=solver.m,
        seconds=seconds,
        status=int(solver.status.numpy()[0]),
        count=int(solver.converged.numpy()[0]),
        finite=finite,
    )
    if finite:
        remainder = vectors - space @ (space.T @ (p.b @ vectors))
        row.update(
            values=values.tolist(),
            expected=expected.tolist(),
            max_residual=float(
                np.linalg.norm(p.a @ vectors - (p.b @ vectors) * values, axis=0).max()
            ),
            orthogonality=float(abs(vectors.T @ (p.b @ vectors) - np.eye(p.k)).max()),
            requested_subspace_error=float(
                np.sqrt(np.maximum(0, np.sum(remainder * (p.b @ remainder), axis=0))).max()
            ),
            eigenvalue_error_scaled_to_weak_component=float(
                abs(np.sort(values) - expected).max() / scale
            ),
        )
    graph = solver.graph = None
    inverse.close()
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="build/weak-block/probe.json")
    args = parser.parse_args()
    rows = []
    for scale in [1e-4, 1e-10, 1e-12]:
        for shift in [0.0, 1e-14, 1e-12, 1e-10]:
            row = run(scale, shift, 1)
            rows.append(row)
            print(row, flush=True)
            Path(args.out).write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
