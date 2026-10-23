# cuDSS lifecycle and source-upgrade regression fixes

Warpack pins optional binding `e1d1be41decb8d6a4305acdd40e26a5a86c0f797`, containing
[warp-cuDSS PR #3](https://github.com/alecjacobson/warp-cuDSS/pull/3). The NVIDIA binary
remains **0.8.0.10**. This addresses [#194](https://github.com/alecjacobson/warpack/issues/194)
and the installation procedure in [#195](https://github.com/alecjacobson/warpack/issues/195).

An invalid reordering name previously allocated three native objects before
raising. Ownership state was initialized only after success, so the destructor
could not release partially constructed objects. A CPU-only mocked-resource probe
records three creates and zero destroys. Setup similarly lost non-null matrix
descriptors returned alongside a create error. Cleanup errors could leave stale
analysis state or repeat destruction of the same pointer.

The binding validates enums before allocation, initializes ownership first,
retains partial outputs, and unwinds construction failures. Explicit release
attempts all owned resources once and reports its first cleanup error afterward.
Failed descriptor cleanup invalidates analysis. A caller whose later `setup()`
fails must still close the solver or use its context manager. Default sparse-solve
settings, factorization calls and numerical arithmetic are unchanged.

| Regression check | Result |
|---|---|
| Original constructor controls | 13 failed, 4 passed |
| Descriptor controls before their fix, after constructor fix | 4 failed, 19 passed |
| Fixed loader/lifecycle host controls | 36 passed |
| Full backend suite, including real supported GPU solves/capture/reuse | 49 passed, 0.75 s |
| Warpack downstream suite, pinned Warp 1.18/FSAI | 137 passed, 118.49 s |
| Focused Warpack compatibility suite, stock Warp 1.15 | 56 passed, 15.43 s |

Injected failures cover constructor creates/configuration, partial matrix creates,
analysis/factorization errors, and cleanup errors. Actual downstream checks cover
capture preparation, changed RHS, refinement, generalized transforms, complex
connections/shifts and damped pencils. No performance improvement is claimed.

## Same-version installation trap

An ordinary pip install of the new Git pin cloned the requested commit but left
the older installed `3f2091f` binding in place; both commits declare version 0.1.0.
The first downstream test attempt was stopped after inspecting installed
`direct_url.json`. Its partial log is retained and is **not validation of the new
binding**. The explicit `--force-reinstall --no-deps` command then installed the
requested revision. Installed provenance and all four package modules were checked
against the merged source before the successful rerun. Neither Warp nor the NVIDIA
binary was replaced, and the original editable source checkout was untouched.

The [installation guide](../../docs/installation.md#updating-an-existing-cudss-installation)
contains the verified command and Git-revision check. Raw failed controls,
installation logs, stopped attempt, actual installed identity and final passing
logs are retained here. GPU checks used the L40/driver 570.158.01 with serialized
GPU access, CPU affinity 8–11 and four OpenMP/OpenBLAS threads.
