# Adjacent RHS storage for the inner CG matrix multiply

Applying the same layout idea as the preceding [FSAI experiment](COALESCED_FSAI.md)
to the matrix multiply reduces the Anvil block solve to **1.8292s**, with an
independent-process replication at **1.8432s**. The preceding FSAI-only profiles
were 1.8976/1.9358s. The new profiles are about **20% faster** than the fresh
2.2915s original-layout control. The earlier cuDSS control is still about 3.9x
faster; issue #15 remains open.

The optional `--matrix-layout columns` keeps upstream Warp CG and the same
matrix/preconditioner. It transposes right-hand sides to adjacent storage,
applies scalar CSR or 3x3 BSR with eight RHS lanes and four sparse-entry lanes,
and transposes back with the alpha/beta update. Both working buffers are
allocated before capture. Changing matrix values in captured buffers is
supported. Input/output aliasing is tested. This option stays in the benchmark
tree, defaults to `rows`, and propagates to every preallocated active CG width
and the adapter supplied to CuPy.

## Component comparison

Identical FP32 Anvil matrix and random inputs, 15 alternating samples per layout
and width. Each output is checked against independent SciPy FP64 accumulation;
all 120 samples pass, maximum relative difference 6.91e-8. Timings include both
transposes and the output update. The new graph contains three kernels and no
copies or host nodes.

| RHS width | Previous rows, ms | Columns including transposes, ms |
|---|---:|---:|
| 4 | **0.1031** | 0.1386 |
| 8 | 0.1964 | **0.1486** |
| 16 | 0.3969 | **0.2696** |
| 24 | 0.5818 | **0.4189** |

Width four remains a slower control. Transposing one matrix application saves
less work than transposing once for both FSAI factors. No universal speedup for
other batch widths, sparsity patterns or GPUs is claimed.

## Complete Anvil comparison

Same 283,857 DOFs, 20 elastic modes and six rigid constraints, FSAI32, shift 100,
FP32 CG25/.01 and FP64 outer arithmetic as the previous report. Warp uses width
24, active CG/search buckets of four and full history; CuPy uses width 20 and
the same captured upstream CG with both layout changes.

| Matrix layout, with column FSAI | Warp median, s | CuPy median, s |
|---|---:|---:|
| Rows, preceding profile | 1.897630 | 2.168628 |
| Rows, preceding replication | 1.935807 | 2.174956 |
| Columns | **1.829175** | 2.168037 |
| Columns, replication | 1.843231 | **2.142116** |

All 12 new complete-solve samples pass every original spectrum/residual,
mode-shape and orthogonality gate. Warp returns 20/20 with status zero; CuPy also
passes its native fresh absolute 1e-5 stopping criterion. Warp's maximum
original residual over both profiles is **1.8213e-8**, and its maximum normalized
returned-pair residual **9.1062e-9**, below the unchanged 1e-8 bound. The stopping
formulas differ. Rounding changes can affect termination: each new Warp profile
has two 17-step/8,000-scheduled-vector-iteration trials and one
16-step/7,800-iteration trial. The medians do not select the faster short trial.
The observed CuPy differences are small and are not claimed as a robust
whole-solve improvement over the FSAI-only version.

Profiles run sequentially with clean occupancy records, not alternating full
solves. Each process independently assembles the same mesh. Setup, capture,
warmup and CPU validation are outside solve timing. The first new profile has
2.0417s setup, including encountered JIT work; replication setup is 1.2185s and
capture 1.1266s. The device free-memory decrease through setup/capture is 4.454GB
(not peak memory), about 67MB more than the FSAI-only adapter. Replay has 30
conditional nodes and 110 DtoD copies, with no HOST/HtoD/DtoH nodes.

The subsequent [settings sweep](ANVIL_SETTINGS.md) retains width 24, FSAI32
and CG25: six tested alternatives do not improve the accepted comparison. Its
fresh unchanged-code control is 1.7784 s Warp / 2.1676 s CuPy.

## Validation and reproduction

**128 tests pass in 7.07s** on the pinned Warp environment. These include
standalone scalar/BSR FP32/FP64 operators, widths crossing tile boundaries,
changed captured matrix values and RHSs, aliases, zero-beta NaN guards,
independent scaled/zero CG RHSs, active-bucket reactivation and final generalized
eigenspaces. All **32 standalone operator tests pass stock Warp 1.15 in 4.21s**.
Lint/format and whitespace checks pass. This is focused validation; no installed
core files changed and the prior full-suite record remains separate.

Add `--matrix-layout columns` to the command in the [FSAI report](COALESCED_FSAI.md)
for the full solve. Run `python -m benchmarks.experimental.compare_operator_layout
--out build/operator-layout.json` for components, under the occupancy monitor.
The previous component baseline's `RepeatedOperator` comes from f7e33ea; the
current row path retains its numerical kernel. Source hashes identify each run.

Raw [components](../../results/performance_followup/coalesced-operator-components.json),
[complete solve](../../results/performance_followup/coalesced-operator-anvil.json),
[replication](../../results/performance_followup/coalesced-operator-repeat.json),
and [validation](../../results/performance_followup/coalesced-operator-validation.json)
are retained with corresponding occupancy records.
