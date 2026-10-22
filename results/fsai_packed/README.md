# Experimental packed Cholesky workspace for FSAI

Follow-up to [the FSAI memory diagnosis](../fsai_memory/README.md) and
[#131](https://github.com/alecjacobson/warpack/issues/131), L40, 2026-10-22.
This checksum-guarded experiment changes only per-row Cholesky workspace from
a square matrix to packed lower-triangular storage. Adaptive support search,
arithmetic, factor packing and application remain upstream Warp.

The installed dependency is unchanged. [Patch](fsai-packed.patch) applies to
FSAI SHA256 `35c0f0cba8318fe74fa6b4b471679fe56a17c22f40331d3c89a2053bf62d9d07`
at revision `04427b997fe7ff07bc6ed1dde5ca9813496c6b55`; candidate SHA256 is
`a5f00cb85a1cc411d836b9edc5f0e64f886e3da3d172cb1a1f5e85b551941492`.
A [separate port to the newer dependency](../fsai_current/README.md) passes
captured refit/CG and changing-FEM tests under
[#139](https://github.com/alecjacobson/warpack/issues/139). The memory measurements
below apply only to the original pinned revision.

## Memory result

Fresh processes construct FP32 FSAI factors with FP64 setup on a 96-row SPD
tridiagonal, growth batch 3. All four memory workers are uncontended. Memory is
the device-free decrease across construction, distinct from live factor bytes.

| Width | Original context/setup decrease (MiB) | Packed (MiB) | Original build local bytes/thread | Packed |
|---|---:|---:|---:|---:|
| 32 | 1,730 | 906 | 9,128 | 5,160 |
| 64 | 13,844 | 7,136 | 67,368 | 35,112 |

Register count remains 48. Deleting factors restores live Warp pool use, while
the device-free decrease remains in the CUDA context. This is evidence of a
smaller setup/context footprint, not a claim that factor buffers leaked or
that every GPU/driver uses identical local-storage backing. Setup seconds in
the raw records are single observations, not a stable timing comparison.

CPU construction/refit tests are bitwise equal for widths 32 and 64. GPU tests
also find identical supports and factor/transpose values after construction and
refit on dense SPD matrices with and without permutation. They compare the
unmodified dependency against the candidate in the same process. Width-64
comparison requires about 14 GiB of context memory and is explicitly opt-in:

```bash
export PYTHONPATH="$PWD/build/warp-fsai:$PWD"
WARPACK_FSAI_LARGE_MEMORY=1 python -m pytest -q tests/test_fsai_packed_storage.py
python -m benchmarks.monitor_gpu --out build/packed64.occupancy.json \
  --log build/packed64.log -- python -m benchmarks.experimental.fsai_packed_storage \
  --width 64 --gpu packed --out build/packed64.json
```

Both width-64 GPU cases passed in the earlier isolated run. A later combined
run with a foreign job using 35 GB failed allocating the original width-64
workspace; the other 111 tests passed. After sufficient memory became available,
the explicit opt-in suite passed all five tests again. The resource failure is
retained in this account, and neither accuracy rerun is a timing measurement.

## FEM validation and timing limits

The four `fem-{upstream,packed}{32,64}.json` records request 20 modes of a
10,890-DOF clamped procedural tetrahedral beam with consistent mass. All pass
native counts/status, original equations, M-orthogonality, requested spectra
and complete subspaces at the common 1e-8 gates; worst subspace error is below
9e-12. Graphs contain no HOST/HtoD/DtoH nodes.

All four final FEM workers are uncontended: one warmup and three accepted
repeats in each fresh process. Warm runtime is the median of the complete
random-start workflow. Setup includes host assembly, factor/solver construction
and capture, and is a single observation with JIT/context effects.

| Width / implementation | Assembly + setup + capture (s) | Warm workflow (s) |
|---|---:|---:|
| 32 / upstream | 1.2149 | 0.1926 |
| 32 / packed | 1.2617 | 0.1981 |
| 64 / upstream | 1.8657 | 0.1923 |
| 64 / packed | 1.8284 | 0.1942 |

Packed warm time is within 3% of the original at both widths, below the 25%
regression budget. The demonstrated benefit is memory, not faster eigensolves.
All physical/native checks pass. This is not an Anvil measurement or a
smaller-GPU portability claim. Earlier contended attempts are retained under
`screening/` and `screening-repeat/` and excluded from this table. The strict
reporter rejects those attempts as performance baselines.

```bash
python -m benchmarks.monitor_gpu --out build/packed-fem.occupancy.json \
  --log build/packed-fem.log -- taskset -c 8-11 \
  python -m benchmarks.experimental.fsai_packed_fem --variant packed \
  --row-size 32 --out build/packed-fem.json
```

Next: complete the current-revision port comparison and test smaller-memory
GPUs. This experiment does not close #131 or alter the released Anvil recipe.
