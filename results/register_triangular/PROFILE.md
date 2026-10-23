# Remaining 200-mode stage costs

The uninstrumented full solve passes before and after profiling. Eight selected
step replays (two at each checkpoint) have zero status. Source/dependency hashes
are current and occupancy is clean. Host readback selects diagnostic instrumentation
only; the original full graph remains device controlled with no HOST/HtoD/DtoH
nodes. These event spans overlap: **do not add columns or infer whole-solve
attribution from their sum**. Conditional first/second-whitening helpers are
measured as complete helpers; their nested canonical sweeps can be stale.

Median milliseconds at each fixed checkpoint, installed 200-mode compact-history
configuration.

| Checkpoint | Active residuals | Selected step | Preconditioner | All Gram calls | X orthogonalization | Search orthogonalization | Ritz eigensolve |
|---:|---:|---:|---:|---:|---:|---:|---:|
|1|204|420.847|57.111|82.721|22.149|91.371|74.217|
|20|60|105.801|4.645|26.896|17.230|25.103|20.042|
|40|9|51.275|1.644|13.061|17.227|5.161|7.704|
|50|5|47.619|1.230|12.394|17.219|3.862|6.798|

At checkpoint 50, the nested triangular span is 5.412 ms and Cholesky 2.205 ms
inside the 17.219 ms X-orthogonalization stage. Early wide projection/Gram and
projected-eigen costs remain substantial. This motivates a distinct Gram-product
screen; it does not prove a particular tile shape will help.

[Raw profile](profile/200-a1.json) and adjacent log/occupancy contain complete
settings, input identity, all events and before/after physical checks. The
[whole-solve CuPy comparison](CUPY.md) remains the acceptance target.
