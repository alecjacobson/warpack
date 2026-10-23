# Verified independent FEM reference reuse (#175)

Optional reference caching reduces benchmark turnaround without changing
backend setup or solve timing. It caches independently computed CPU eigenvalues
and vectors; each worker still assembles its current matrices, applies current
boundary-cluster/nullspace policy, and builds its own factor or preconditioner.
The cache is outside `warpack/` and uses no solver output as a reference.

An explicit wrapper is available for the first-whitening comparisons:

```bash
python -m benchmarks.experimental.compare_cached_first_whitening \
  --reference-cache /path/to/external/reference-cache \
  --first-whitening padded \
  ...the usual compare_first_whitening arguments...
```

Other isolated FEM workers can use `FEMReferenceCache(path)` as a context around
fixture construction. This context temporarily wraps the CPU reference function
and is intended for a sequential worker. Existing benchmark entry points do
not silently opt in. Anvil retains its established separate fixture protocol.

The key contains exact canonical CSR content/shape/dtype, reference count, full
original reference-function source (including all fixed options and seed),
fixture/policy/helper source hashes and NumPy/SciPy versions. Matrix content and
count define the reference input; fixture parameters remain in the enclosing
benchmark report. All guard eigenvectors are stored. Cluster extension and
known rigid-space construction execute fresh after loading.

Entries contain numeric NPZ arrays and a JSON manifest. Loading forbids pickle,
verifies payload and array hashes/provenance, then rechecks original equations,
capacity, ordering and mass orthogonality against the current matrices. A corrupt
or numerically invalid entry fails closed; it is never silently regenerated or
overwritten. Per-key file locks serialize writers; a private staging directory
is renamed only after complete validation. Reports distinguish creation from
verified reuse and include exact identities and CPU preparation time. All cache
work is outside timed backend setup/capture/replay.

## Qualification

Four Xeon 8362 cores, 2026-10-23; single creation/reuse observations for the
34,560-unknown clamped consistent-mass fixture, including fresh assembly and
fixture policy. These measure benchmark preparation, **not solver speed**.

| Requested modes | Fresh fixture + reference s | Fresh fixture + verified reference reuse s |
|---|---:|---:|
|100|40.9155|0.8658|
|200|70.9635|1.1450|

Full reference identities, including guard values and requested vectors, match
both across creation/reuse and the earlier uncached paired-Jacobi controls
exactly. [Raw qualification](qualification.json) retains complete identities,
provenance and observations. Large cache arrays live outside the repository.

**29 CPU tests pass**: parameter/content/source/dependency invalidation, numeric
checks on every read, corruption and partial entries, pickle/object rejection,
thread/process concurrency, actual sparse-matrix changes, ARPACK guard capacity,
and six FEM families including repeated boundary clusters and rigid nullspaces.
[Transcript](validation/tests.log).

The actual wrapper also passes **six complete first/later outputs** in small
canonical/padded solves, with identical input/source/dependency identities and
creation followed by verified reuse of the same reference. All native and
independent physical gates pass. [Smoke reports](smoke/) retain these checks;
their backend timings are not performance claims because those workers did not
run under the external occupancy monitor.
