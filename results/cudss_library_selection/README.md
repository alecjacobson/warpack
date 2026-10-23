# Reliable cuDSS library selection

The current checkout pins warp-cuDSS `3f2091fb1ae156609c38122dbb0a1d3c83913f74`,
including [explicit-path error handling](https://github.com/alecjacobson/warp-cuDSS/pull/1)
and [current pip-layout discovery](https://github.com/alecjacobson/warp-cuDSS/pull/2).
The NVIDIA binary remains **0.8.0.10**. This resolves Warpack issues
[#190](https://github.com/alecjacobson/warpack/issues/190) and
[#191](https://github.com/alecjacobson/warpack/issues/191).

Previously, an explicit path that failed to load could silently select another
library. Standalone discovery also missed the supported wheel's `nvidia/cu12/lib`
layout. Warpack compensated for the latter by setting `CUDSS_LIBRARY_PATH` during
inverse construction. Discovery now belongs to the optional binding, and Warpack
no longer changes the environment. A nonempty explicit override must load
successfully, including its dependencies. The first successful library remains
cached for the process lifetime, preserving live factors and graph ownership.

## Regression evidence

| Check | Result |
|---|---|
| Original binding, explicit-selection regressions | 2 failed, 5 passed |
| Fixed explicit selection, host-only | 7 passed |
| Initial standalone full suite, before pip-layout fix | 13 failed, 7 passed (library loading) |
| Explicit supported binary, before pip-layout fix | 20 passed |
| Both loader fixes, host-only | 13 passed |
| Both fixes, standalone full suite without override | 26 passed |
| Warpack integration, pinned Warp 1.18/FSAI environment | 56 passed in 15.39 s |
| Same integration checks, stock Warp 1.15 | 56 passed in 31.08 s |

Warpack checks cover fresh-process absent/empty overrides without environment
mutation; invalid explicit paths without fallback; changed-RHS captured solves;
first-capture preparation; indefinite refinement; generalized transforms; and
complex shift refresh. Original numerical tolerances remain unchanged. These are
correctness checks, not performance measurements. Source changes are limited to
loading/setup; eigensolver arithmetic and sparse-solve configuration are unchanged.

The first integration launcher exposed the backend checkout's `tests` package on
`PYTHONPATH`, shadowing Warpack's test namespace, and failed collection. The retained
`cudss-loader-integration-tests.log` records this harness failure. The corrected
launcher uses an isolated copy of only the backend package, with every Python
module byte-checked against the merged source; it runs the identical test list.
The actual environment was subsequently installed from the pinned Git revision.
`cudss-loader-installed-identity.json` records its installed path and Git provenance.
The existing editable backend source checkout was left untouched.

`cudss-loader-identity.json` records the actual loaded binary path, SHA256 and
runtime-reported version **0.8.0**, rather than assuming package metadata identifies
the loaded library. The supported binary SHA256 is
`0740250950abbbbc7690c4edb1506f8432558658764d6c1f7cba29615acb3dfc`.
The binding source SHA256 is
`747914f904e4c4fb491189932ba9035a0214eeaf41e88e243ee2f7eaafa66e84`.

All runs used the existing L40, driver 570.158.01, CPU affinity 8–11 and four
OpenMP/OpenBLAS threads. GPU regression runs were serialized under the shared
GPU-work lock. Raw logs retain failed discovery and collection attempts as well
as passing runs. cuDSS 1.0 qualification remains separate in
[#189](https://github.com/alecjacobson/warpack/issues/189); this change does not
claim to remove cuDSS 0.8's captured vendor metadata transfers (#29).
