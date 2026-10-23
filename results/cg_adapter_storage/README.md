# Active-CG adapter storage (#166)

The allocation inventory uses an actual diagonal 3×3 block system with
n=34,560, width 204, buckets 4..204, FP32 working arithmetic and column layouts.
This is a shape-matched allocation diagnostic, not a complete FEM solve or a
performance comparison. Sparse matrices/factors, outer eigensolver storage,
runtime/pools and peak memory are outside its named scratch counts.

Before adapter sharing, actual named spans are:

| Category | Bytes |
|---|---:|
| Adapter right-hand side / solution | 1,466,449,920 |
| Sparse matrix application | 1,466,449,920 |
| FSAI factor application | 1,466,449,920 |
| Upstream CG internal vector buffers | 2,932,899,840 |
| Outer packing (already shared by #165) | 112,803,840 |

The first three categories total **4,399,349,760 bytes**. The new allocation policy shares those six adapter-owned arrays across mutually
exclusive buckets, reducing them to 169,205,760 bytes and saving 4,230,144,000 bytes
(~3.94 GiB), confirmed by the complete controls below. Upstream CG's internal
vectors and numerical implementation remain independent.

[Raw inventory and provenance](before-inventory.json) includes control/reduction
arrays and a deduplicated named total. Private upstream fields are read only for
this diagnostic; no upstream solver state is patched. Cross-category aliases are
deduplicated in the total, so adding category totals need not equal that total.

## Qualified complete controls

L40, pinned Warp/FSAI, one independently validated first output and three measured subsequent replays per fresh worker. All 24 complete outputs pass the unchanged physical/spectrum/subspace and native stopping/count gates. All six external occupancy reports are clean. Each separate/shared pair has identical exact inputs, source manifests and dependency fingerprints. Outer packing is shared in both controls, and compact projected graphs are disabled in both.

| Problem | Exact requested adapter / observed live-pool saving, bytes | Live Warp pool after capture, separate → shared, bytes | Replay median, separate → shared, s |
|---|---:|---:|---:|
| [anvil](full/anvil-shared-a1.json) | 408,754,080 | 2,726,317,020 → 2,317,562,940 | 1.6190 → 1.6134 |
| [100](full/100-shared-a1.json) | 1,078,272,000 | 2,922,845,852 → 1,844,573,852 | 3.0292 → 3.0252 |
| [200](full/200-shared-a1.json) | 4,230,144,000 | 10,277,063,180 → 6,046,919,180 | 10.4632 → 10.4511 |

The measured live-pool reduction exactly matches the requested backing-storage saving in every pair. Upstream internal CG vector byte counts are unchanged. Device-free-memory decreases, setup/capture/first-solve phases and every replay sample are separately recorded in the raw JSON. These pool counts are not total context memory or peak measurements. Replay differences are below 0.4%; no arithmetic speedup is claimed.

Combined with the earlier [packing-buffer change](../cg_packing/README.md), the 200-mode fixture saves 7,050,240,000 requested bytes (~6.57 GiB). Its recorded live Warp pool falls from 13,097,159,180 bytes before that change to 6,046,919,180 bytes here. This does not qualify a smaller GPU or remove upstream FSAI setup requirements.

## Implementation and validation

The private active-bucket factory supplies contiguous prefix views of the maximum-width base arrays to Warpack adapters. Upstream CG receives those caller-owned b/x arrays through its normal constructor; its internal workspaces and implementation are untouched. Matrix/factor application kernels, arithmetic, zero initialization, matrix refresh, active selection, counters and stopping are unchanged. Public `for_width` still returns independent adapter buffers, and class overrides of that factory retain their behavior.

All **193 affected pinned tests pass, with 1 skip**. Stock Warp passes 68 with 126 upstream-feature skips; it does not support this batched-CG path. Expanded coverage exercises all supported row/column matrix/factor layouts, with/without FSAI, scalar/3×3 matrices, FP32/FP64 work, in-place and changed/scaled RHSs, changed matrix values, shrinking/growing/nonleading selections, actual backing aliases, contiguous views, independent upstream vectors, independent public-width buffers and custom factories. Complete eigensolver and changed-input replay tests remain included.

The clean wheel has 39 core modules matching source byte-for-byte, excludes benchmarks/examples/tests, and passes isolated changed-matrix replay with SciPy, CuPy, benchmark/example and optional cuDSS imports forbidden. Logs and package evidence are retained here.

Review: only one device-selected branch uses the shared backing at a time. Every branch copies/zeros its current working RHS/solution and overwrites its complete application scratch. Flat-prefix reshaping gives each column-layout branch its own contiguous logical width without preserving a larger parent stride. Source preservation for in-place outer application remains separate. Graph buffer addresses remain fixed; new storage selection runs only during construction, with no replay allocation or host selection. Internal CG reduction/search vectors and branch controls remain independently allocated.

Reproduce with `python -m benchmarks.experimental.compare_cg_adapter_storage --adapter-storage separate|shared --public-api ...`; exact remaining arguments and provenance are recorded in each JSON. The private separate switch is for same-source regression controls. Shared adapter storage is the installed default.
