"""Benchmark-only initial-guess experiments using the unchanged upstream CG state."""

import argparse
import json
from pathlib import Path

import warp as wp

from warpack import GeneralizedKrylovSchur
from warpack.generalized import accumulate_cg_statistics

from . import compare
from .hodge import make_hodge_case


@wp.kernel
def guess_dots(b: wp.array[wp.float64], ab: wp.array[wp.float64], sums: wp.array[wp.float64]):
    chunk = wp.tid()
    x = wp.tile_load(b, shape=256, offset=chunk * 256)
    ax = wp.tile_load(ab, shape=256, offset=chunk * 256)
    wp.tile_atomic_add(sums, wp.tile_sum(wp.tile_map(wp.mul, x, x)), offset=0)
    wp.tile_atomic_add(sums, wp.tile_sum(wp.tile_map(wp.mul, x, ax)), offset=1)
    wp.tile_atomic_add(sums, wp.tile_sum(wp.tile_map(wp.mul, ax, ax)), offset=2)


@wp.kernel
def set_guess(
    b: wp.array[wp.float64], x: wp.array[wp.float64], sums: wp.array[wp.float64], mode: int
):
    i = wp.tid()
    numerator, denominator = wp.float64(sums[0]), wp.float64(sums[1])
    if mode == 1:
        numerator, denominator = wp.float64(sums[1]), wp.float64(sums[2])
    scale = wp.float64(0.0)
    if denominator > wp.float64(0.0) and wp.isfinite(numerator) and wp.isfinite(denominator):
        scale = numerator / denominator
    x[i] = scale * b[i]


class Guess:
    def __init__(self, inverse, mode):
        self.inverse, self.mode = inverse, mode
        # Deliberate benchmark-only access; no change to upstream CG or FSAI.
        self.operator = inverse.state._A
        self.product = wp.zeros_like(inverse.b)
        self.sums = wp.zeros(3, dtype=wp.float64, device=inverse.device)

    def apply(self, x, y):
        inv = self.inverse
        for j in range(x.shape[0]):
            self.operator.matvec(x[j], self.product, self.product, alpha=1.0, beta=0.0)
            self.sums.zero_()
            wp.launch_tiled(
                guess_dots,
                dim=(inv.n + 255) // 256,
                inputs=[x[j], self.product, self.sums],
                block_dim=128,
                device=inv.device,
            )
            wp.launch(set_guess, inv.n, [x[j], y[j], self.sums, self.mode], device=inv.device)
            iterations, residual_sq, tolerance_sq = inv.state(b=x[j], x=y[j])
            wp.launch(
                accumulate_cg_statistics,
                1,
                [iterations, residual_sq, tolerance_sq, inv.stats],
                device=inv.device,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["zero", "rayleigh", "minres"], default="rayleigh")
    parser.add_argument("--method", choices=["warp-cg", "warp-cg-fsai"], default="warp-cg-fsai")
    parser.add_argument("--iteration", choices=["block", "krylov"], default="block")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--ncv", type=int, default=32)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    original = compare.warp_setup

    def setup(*setup_args, **kwargs):
        solver, inverse = original(*setup_args, **kwargs)
        if args.iteration == "krylov":
            p = setup_args[0]
            solver = GeneralizedKrylovSchur(
                solver.op, solver.mass, inverse, p.k, ncv=solver.m, tol=solver.tol, target=p.target
            )
        if args.variant != "zero":
            guess = Guess(inverse, int(args.variant == "minres"))
            inverse.apply = guess.apply
        return solver, inverse

    compare.warp_setup = setup
    p = make_hodge_case("large_disconnected", 41).problem
    row = compare.run("D", p.n, args.method, ncv=args.ncv, steps=args.steps, problem=p)
    row.update(
        initial_guess=args.variant,
        iteration=args.iteration,
        diagnostic_only=True,
        note="CG recurrence and preconditioner are unchanged upstream Warp; initial guess is a scalar multiple of the current RHS, not recycled state or a known-nullspace projection",
    )
    Path(args.out).write_text(json.dumps(row, indent=2) + "\n")
    print(
        args.variant,
        args.method,
        row["outcome"],
        row.get("solve_seconds"),
        row.get("cg_statistics"),
        flush=True,
    )


if __name__ == "__main__":
    main()
