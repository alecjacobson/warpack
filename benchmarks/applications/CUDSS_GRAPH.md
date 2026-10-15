# cuDSS graph transfers and host API audit

[Issue #29](https://github.com/alecjacobson/warpack/issues/29) remains open for
eliminating vendor host-to-device metadata copies. The audit finds **no explicit
CUDA synchronization API calls during warm replay**. It also measures host
enqueue overhead, so absence of a synchronization call must not be read as
zero host-side cost or a guarantee that launch never experiences backpressure.

Production numerical code and defaults are unchanged. The strict
`device_only_graph` gate remains false for the audited cuDSS graphs.

## Observations

The [isolated API trace](../../results/cudss_scalar_probe.json) covers SPD,
symmetric and general matrix types, with cuDSS refinement and Warpack correction
both disabled. Each solve captures two eight-byte HtoD copies. CUPTI observes
those copies during eager solve/capture. Warm replay calls only
`cudaGraphLaunch` and its underlying `cuGraphLaunch`; the explicit caller wait
is recorded separately as `cuCtxSynchronize`. First replay also instantiates and
uploads the graph. No convergence values are read back by the solve.

The copied bytes decode as two 32-bit integers `[0, n-1]` for n=4, 16 and 64.
They remain unchanged across three different RHS blocks on diagonal, chain and
dense matrices, including eight RHSs. The large n=10,890 solve copies
`[0, 10889]`. This is **consistent with row-work-range initialization**, an
inference from the observed payloads, not a documented cuDSS internal contract.
The source addresses stay valid while the owning inverse is retained in these
experiments. Graphs must be destroyed before releasing the inverse; this audit
does not establish lifetime beyond that ownership boundary.

The [configuration probe](../../results/cudss_config_probe.json) verifies actual
settings and correct solutions. Explicit zero refinement, explicit general
solve, deterministic mode and disabled superpanels all retain the two copies.
NVIDIA documents default solve as currently equivalent to general solve, and
refinement defaults to zero. Hybrid memory/execute are disabled here. See
[NVIDIA's configuration reference](https://docs.nvidia.com/cuda/cudss/types.html).
Analysis is setup work; NVIDIA documents factorization/solve as asynchronous
under the tested non-hybrid conditions in its
[graph support description](https://docs.nvidia.com/cuda/cudss/general.html#cuda-graphs-support).
No tested supported setting removes the metadata transfers.

## Large FEM measurements

The [large audit](../../results/cudss_large_audit.json) uses the 10,890-DOF
consistent-mass clamped beam, k=6, ncv=32 and one fixed Krylov cycle. CUPTI is
**disabled during timing measurements**. Three synchronized warm trials per
profile pass original residual, full requested subspace, mass orthogonality and
strict 6/6 counts, with no observed competing GPU process.

| cuDSS profile | Replay median | HtoD nodes |
|---|---:|---:|
| Default | 17.479 ms | 64 |
| Explicit zero refinement | 17.715 ms | 64 |
| Explicit general solve | 17.454 ms | 64 |
| Deterministic | 28.528 ms | 64 |
| Superpanels disabled | 17.301 ms | 64 |

These measurements do not justify changing defaults. The small differences
among the non-deterministic settings should not be treated as robust tuning
gains. The saved default FEM timing/accuracy/graph gate passes at **0.99987×**.
No Anvil result or cuDSS version upgrade is implied.

The full captured FEM graph has 64 HtoD nodes (512 bytes total), 98 DtoD nodes,
no host callback nodes and no DtoH nodes. Its warm API trace again contains only
three runtime graph launches and the three corresponding driver calls. The
pure-Warp CG control has no host transfers/callbacks, only three DtoD copies;
its maximum original linear residual is `1.10e-13`.

Uninstrumented FEM launch calls take about **0.38 ms** when each replay is
completed before the next. A three-launch queued diagnostic takes about
0.38/0.36/1.99 ms to enqueue, while all three graphs finish in 52.37 ms. Thus
API traces without a synchronization call do not establish negligible host
enqueue cost. The CG control launches in roughly 9–13 microseconds, but it has
a different graph/workload and is not a causal comparison of transfer cost.

## Diagnostic source-residency control

The [fixed-pencil control](../../results/cudss_copy_control.json) changes only
the source residency of the 64 metadata copies: identical bytes are initialized
in owned Warp device arrays, then substituted into an uninstantiated diagnostic
graph. All kernels, copy-node counts, factors, starts, budgets and destinations
are retained. The control therefore has 162 DtoD nodes and no host transfers.
Every individually completed trial and each final queued output passes the
same numerical checks and strict counts. Queued intermediate outputs are not
individually validated and are labeled diagnostic.

| Fixed captured graph | Median enqueue | Median completion | Third queued enqueue |
|---|---:|---:|---:|
| Original vendor sources | 0.363 ms | 17.460 ms | 2.005 ms |
| Device-source diagnostic | 0.284 ms | 17.449 ms | 0.271 ms |

Source residency affects host enqueue/backpressure in this experiment, while
complete replay time changes by only about 0.06%, too little to claim a robust
throughput gain. The metadata copies do not explain a large eigensolve timing
gap on this fixture.

This control edits **private vendor graph nodes** and is not a supported
production backend. Its fixed metadata assumption is not established across
other cuDSS versions, configuration changes, refactorizations or source-value
updates. No such rewrite is enabled by Warpack. A supported device-source path
or vendor confirmation/change remains work in #29; the benchmark does not
convert the production graph contract into a pass.

## Validation and reproduction

Audited environment: L40, Warp 1.18.0+cu12, driver 570.158.01, cuDSS 0.8.0.10,
CUDA bindings 12.9.7 and CUPTI 12.8.90. Use the pinned Warp/FSAI environment from
[FEM.md](FEM.md). Instrumentation is optional benchmark code and is never
imported by production Warpack.

```bash
python -m pip install cuda-bindings==12.9.7 nvidia-cuda-cupti-cu12==12.8.90
export PYTHONPATH="$PWD/build/warp-fsai:$PWD"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python -m benchmarks.applications.cudss_scalar_probe
python -m benchmarks.applications.cudss_config_probe
python -m benchmarks.applications.cudss_large_audit
# Explicit diagnostic-only graph rewriting experiment:
python -m benchmarks.applications.cudss_copy_control
```

The full numerical suite's preceding checkpoint is **311 passing tests**.
This change adds audit tools and evidence, with no production numerical edits.
The audits validate three isolated matrix types, eleven configuration/shape
profiles, five large timing profiles, and the pure-Warp/control graphs. Lint,
formatting and wheel build pass. See [source and validation provenance](../../results/cudss_graph_validation.json).

An initial instrumentation attempt failed at runtime node-type introspection
on a conditional graph after completing the five valid timing profiles. The
final tools use driver-generated DOT for full-graph source inspection and
scoped CUPTI subscriptions for cleanup. Those timing samples were retained;
traces and enqueue measurements were rerun separately. DOT and API tracing
answer different questions: neither alone measures every possible internal
driver stall or guarantees behavior for untested configurations.
