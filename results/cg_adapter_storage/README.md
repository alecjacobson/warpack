# Active-CG adapter storage (#166)

The allocation inventory uses an actual diagonal 3×3 block system with
n=34,560, width204, buckets4..204, FP32 working arithmetic and column layouts.
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

The first three categories total **4,399,349,760 bytes**. Sharing those six
adapter-owned arrays across mutually exclusive buckets could reduce them to
169,205,760 bytes, saving 4,230,144,000 bytes (~3.94 GiB). These are actual
before counts and an estimated saving, pending validation of the new allocation
policy. Upstream CG's internal vectors and numerical implementation are separate.

[Raw inventory and provenance](before-inventory.json) includes control/reduction
arrays and a deduplicated named total. Private upstream fields are read only for
this diagnostic; no upstream solver state is patched. Cross-category aliases are
deduplicated in the total, so adding category totals need not equal that total.
