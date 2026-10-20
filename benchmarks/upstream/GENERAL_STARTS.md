# Replayable nonsymmetric starting vectors

`GeneralEigensolver` and both paths of `ShiftInvertEigensolver` accept
`initial_vector`: a finite nonzero device array of shape `(n,)` and dtype
`wp.vec2d`, with real and imaginary components. This completes the single-vector
nonsymmetric start interface tracked in
[#107](https://github.com/alecjacobson/warpack/issues/107) and
[#17](https://github.com/alecjacobson/warpack/issues/17).

```python
solver = GeneralEigensolver(A, k=10, ncv=40, initial_vector=start)
graph = solver.capture(300)
# Update start in place before replay; its current contents are read on device.
wp.capture_launch(graph)

shifted = ShiftInvertEigensolver(
    A,
    k=10,
    sigma=0.2 + 0.3j,
    ncv=40,
    initial_vector=start,
    backend="cudss",
)
shifted_graph = shifted.capture(30)
wp.capture_launch(shifted_graph)
```

Metadata checks reject incompatible shape, dtype, or device before solving.
Initialization reduces the largest finite component, scales before computing
the squared norm, then normalizes in Warp. It neither reads values back to the
host nor changes the supplied array. Default random initialization is unchanged;
when no start is supplied, no new kernels or buffers are added to the captured
solve. cuDSS retains its fixed outer-capture requirement.

Zero and nonfinite starts set nonzero `status` and zero the general convergence
count. An invalid start is replaced internally with a safe work vector so that
the graph can complete without contaminating its reusable state; the resulting
modes must not be used. The error is reapplied after the projected Schur solve,
which otherwise overwrites its own status. The shifted wrapper also clears its
physical counter on invalid starts. Copying a valid start into the same array
restores successful execution on the next replay.

The regression tests cover complex phase changes, finite scales from 1e-320 to
1e300, preserved input contents, normalized initialization, restart-state reset,
direct and cuDSS/GMRES solves, original residuals and nearest-mode selection,
invalid-to-valid replay, and graph transfer/callback checks. The original
Spectra `benchmark_gen_24` matrix and v0 additionally pass a requested-spectrum
check at the explicit width-40 profile. Direct and GMRES graphs contain no host
callbacks or host/device transfers; the cuDSS graph retains the separately
tracked vendor metadata transfers in
[#29](https://github.com/alecjacobson/warpack/issues/29).

This does not resolve the general requested-spectrum limitation in
[#104](https://github.com/alecjacobson/warpack/issues/104). A starting vector,
including a previous eigenvector, need not span the desired invariant subspace.
Always inspect status, residual counts and requested-spectrum coverage. The
historical Spectra benchmark adapter remains available to reproduce its exact
unscaled initialization; its results are not silently replaced by this API.
The previously enabled CuPy complex-v0 cases are Hermitian and were already
covered; no new CuPy general-eigs cases are claimed.

Validation: **159 related tests pass** in the pinned Warp 1.18 environment
(31.73 s) and stock Warp 1.15 (36.55 s), including 15 new tests and replacement
of the obsolete nonsymmetric-start rejection. These are focused runs, not a
rerun of the full suite. No performance claim is made for this API addition.
[Validation record](../../results/performance_followup/general-public-start-validation.json).
