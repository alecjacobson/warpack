# Public block API and CuPy LOBPCG fixtures

The public `warpack.BlockLOBPCG` implements FP64 symmetric smallest-algebraic
block iteration, including SPD coupled mass, supplied device starts and
invariant-space constraints. Its numerical implementation now lives in
`warpack/block.py` and `warpack/block_products.py`. Benchmark-only batched
sparse products, the batched inner-CG adapter and CuPy boundary remain outside
the installed package. Existing Krylov defaults have not changed.

The pinned CuPy LOBPCG family now has **54 passes, 1 failure and 533 explicit
unsupported skips**, replacing its previous blanket 588 skips. This is a
focused run of revision `2316a02d70c7e39d6a7c035529df8358d794256e`, not a full
upstream rerun. [Raw results](../../results/performance_followup/cupy-block-api.json).
48 passes cover dense/sparse FP64 diagonal generalized problems, supplied
preconditioners, non-normalized invariant constraints and COO/CSR/CSC boundary
materialization. Six pass identity/Fiedler cases. 528 skips are unsupported
precision combinations; two require a full-space dense fallback, two use FP32
starts/matrices, and one requests iteration history. The initial adapter report
is retained separately: unsupported cases skipped *inside* CuPy's comparison
decorator appeared as errors, so the plugin now marks them during collection.

The adapter preserves upstream assertions and additionally checks full device
counts, finite status, original-pencil residuals, M-orthogonality, the requested
spectrum and its complete boundary eigenspace. Matrix-backed LinearOperators
and input containers are materialized during test setup. `B` is the mass;
`M` is the supplied preconditioner. No cuDSS or reference eigenvectors are
inserted into this block solve. Largest targets map to smallest of `-A`.
The internal normalized tolerance is `min(upstream_tol, 1e-10)` (or `1e-10`
when omitted). Width equals the upstream starting-block width. Explicit
upstream budgets are retained; omitted budgets mean 20 correction steps.
These differ from CuPy's absolute stopping criterion and iteration semantics.

## The remaining Fiedler comparison

`test_fiedler_large_12` requests the lowest two modes but starts from the exact
highest two modes. Native CuPy and SciPy both return **3.7320508075688767 and
3.931851652578137**, with residuals around `2e-15`. The actual requested values
are **0 and 0.06814834742186336**. Their agreement passes the unmodified upstream
assertion without certifying the requested spectrum.

Warp reaches the correct values after its 20-step budget, but its residuals
`5.22e-7` and `9.52e-7` fail the adapter's `1e-10` criterion: count 0/2. With a
separately declared budget of 100, it converges both pairs in 34 steps. Explicit
`initial_perturbation=1e-3` converges them in 30 steps. Returning this correct
spectrum still cannot pass the original equality assertion against SciPy's
high modes. We retain the failure instead of matching the wrong spectrum or
claiming all upstream tests pass. [Diagnostic](../../results/performance_followup/fiedler-invariant-starts.json),
[tracking issue #72](https://github.com/alecjacobson/warpack/issues/72),
[upstream report](https://github.com/cupy/cupy/issues/10403).

## Reproduce

```bash
export PYTHONPATH="$PWD/build/warp-fsai:$PWD"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
WARPACK_UPSTREAM_REPORT=build/cupy-block-api.json \
  python -m pytest -q -p benchmarks.upstream.cupy_plugin \
  build/test_upstream_cupy_linalg.py -k LOBPCG
python -m benchmarks.experimental.mode_crossings --out build/mode-crossings.json
python -m examples.block_reuse
```

The first command intentionally retains the one failure. The crossing driver
also checks exact/computed full warm blocks after diagonal mode crossings and
records native controls with no perturbation. The example assembles a clamped
P1 bar in Warp, replays on changed stiffness, and copies computed blocks on
device; independent SciPy validation occurs outside capture.
