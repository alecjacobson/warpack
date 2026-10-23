# Where compact-history 200-mode time remains (#147, #173)

Diagnostic fixed-prefix profiles on the same 34,560-DOF beam, width 204,
compact history, pinned Warp/FSAI, L40, four Xeon 8362 cores, 2026-10-23.
The uninstrumented full solve passes all native/original/requested-space gates
both before and after profiling (57/58 outer iterations). Occupancy is clean.

Each row is the median of three individually selected steps from a fixed
checkpoint. **Nested event spans overlap and must not be added.** These are
not adaptive whole-solve timings. Host readback selects diagnostic branches;
the ordinary full solve remains device controlled and its graph audit has no
host nodes and only device-to-device copies.

| Checkpoint | Active residual rows | Selected step ms | CG preconditioner ms | X orthogonalization ms | First metric eigensolve ms | Ritz eigensolve ms |
|---|---:|---:|---:|---:|---:|---:|
|1|204|458.181|57.977|27.298|40.274|83.114|
|5|204|402.971|14.967|21.580|47.750|74.109|
|10|161|310.962|12.066|22.002|32.907|57.916|
|20|60|116.510|4.650|21.524|6.734|20.179|
|35|16|62.210|1.686|21.515|1.362|10.280|
|50|5|51.894|1.232|21.510|0.389|6.819|

Compaction shrinks the residual/search work; it does not shrink the entire
current Ritz basis X. Its orthogonalization stays around 21.5 ms late in the
solve, including approximately 2.21 ms in Cholesky and 9.71 ms in triangular
applications. Those subspans are already inside the X stage. This motivates
[#173](https://github.com/alecjacobson/warpack/issues/173), separately from the
first-metric certificate (#171) and projected Jacobi data reuse (#172).

No pass is removed or tolerance weakened based on this profile. It identifies
where to investigate next; projected eigen solves are not the only remaining
cost. Raw JSON retains every nested stage/sample, source/dependency identities,
fixture metadata and before/after accuracy checks. Use the exact occupancy
command with `benchmarks.experimental.profile_active_block` to reproduce.
