# cuDSS capture setup — issue #132

[Issue](https://github.com/alecjacobson/warpack/issues/132) ·
[Fresh six-backend comparison](../anvil_showcase_132/README.md) ·
[Exact phase medians and performance gates](summary.json)

The original gap was not a slower cuDSS factorization. Both outer solvers use
`CuDSSInverse`. Warp's `capture()` additionally performed a complete eager
Krylov eigensolve to warm kernels before recording the graph. That work was
charged to setup; the benchmark also warmed the completed graph before timing
replays. CuPy creates its inverse operator during setup and enters `eigsh`
during the separately reported first solve/warmup.

## Measured improvement

Three independent fresh-process setups and 15 warm solves per configuration,
L40, four pinned CPU cores, unchanged Anvil accuracy gates. The control forces
the previous full-warmup behavior on the same code and environment.

| Phase | Full-warmup control (s) | Optimized Warp (s) |
|---|---:|---:|
| cuDSS factor construction | 2.1109 | 2.1246 |
| Solver workspace/start upload | 0.0063 | 0.0065 |
| Preparation within capture | 0.6259 | 0.0219 |
| Total capture, including preparation | 0.6714 | 0.2174 |
| Total setup | 2.7887 | 2.3480 |
| Warm solve | 0.4661 | 0.4663 |

Setup falls **15.8%**, with unchanged warm replay speed.
Preparation is a subset of capture, not an additional cost. Phase medians are
computed independently. The local performance gate requires at least 10% lower
setup and no more than 3% replay slowdown; both pass. All original-problem
accuracy checks pass. No ineligible occupancy record enters these medians.

CuPy setup is **2.1151 s**, of which **2.1111 s** is cuDSS construction.
Optimized Warp's factor time is **2.1246 s**. The remaining setup difference
is graph/module preparation, not a different factorization. Fresh-process logs
show about 0.12 s loading the cached Warp tile-product module; recording and
instantiating the graph and priming cuDSS account for the rest. Module loading
moves from the eager warmup into capture, so compare total capture rather than
only the preparation subphase. CuPy also initializes kernels/resources in its
first `eigsh` call; the common table excludes outer warmup for every backend.

## Implementation and regression

For the exact built-in `KrylovSchur` and `CuDSSInverse` types, with no finalizer
and a CUDA driver >=12.3, prime the inverse with one zero-RHS application.
Use eigensolver-owned scratch, preserving starts that alias the inverse's own
`x` or `b`. Keep the synchronization before capture required by the graph-upload
workaround. Supported CUDA drivers load Warp kernels while recording the graph.
The captured numerical iteration is unchanged. Custom classes/operators,
callbacks and older drivers keep the previous full warmup.

The [13 focused capture cases](../../tests/test_cudss_capture_setup.py) check:
- Exactly one eager inverse application before first capture.
- Two projected-space sizes, original residuals, orthogonality, requested
  spectrum, refinement, and changed starts on repeated replay.
- Symmetric-indefinite and general factors, including matching.
- Starting vectors stored in the inverse's `x` or `b` buffers.
- Equal graph node inventories versus the full-warmup control and no HOST nodes
  (the known vendor HtoD metadata transfers remain).
- Conservative fallback for older drivers, custom operators/solver subclasses
  and finalizers that inspect initialized eigenpairs.

Validation: **1,475 passed, 1 skipped** in the full pinned-runtime suite before
the final scratch-alias guard. After that guard and five additional capture
cases, **71 passed on the pinned runtime and 71 on stock Warp 1.15.0**, covering
capture, symmetric/inverse/shift/cycle and multiplicity behavior. Final phase
comparisons and the six-row refresh follow the guard. [Counts, scope and hashes](validation.json),
[full log](full-suite.txt), [final pinned log](final-pinned-regression.txt),
[final stock log](final-stock-regression.txt).

## Reproduction and raw evidence

Use the environment and pinning in the
[fresh comparison recipe](../anvil_showcase_132/README.md#environment-and-reproduction).
Run each in a separate process under `benchmarks.monitor_gpu`; repeat three times:

```bash
python -m benchmarks.anvil_direct --backend warp --full-warmup --repeats 5 --out build/control.json
python -m benchmarks.anvil_direct --backend warp --repeats 5 --out build/optimized.json
```

The flag changes only preparation, providing a control without reverting code.
Phase timing uses synchronized measurements around the actual setup operations.
The setup harness releases its profiling overrides and graph/solver closures
before closing cuDSS resources.

Accepted controls: [1](control-1.json), [2](control-2.json), [3](control-3.json).
Optimized runs: [1](../anvil_showcase_132/warp-direct-1.json),
[2](../anvil_showcase_132/warp-direct-2.json), [3](../anvil_showcase_132/warp-direct-3.json).
Each has an adjacent `.occupancy.json` recording commands and GPU sampling.
Initial probes that overlapped unrelated GPU work remain under `rejected/`;
preliminary accepted measurements before the scratch-alias guard remain under
`pre-alias-guard/`. `after.json` is the initial clean exploratory probe, not an
input to the final medians. No rejected or preliminary result is silently used
as the final result.
