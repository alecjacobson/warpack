# cuDSS workspace ownership on setup failure (#197)

Installing Warpack's allocator destroys the solver's empty native data object
and creates a replacement. If allocator registration or replacement creation
failed, the old implementation retained the destroyed pointer. A failed create
that returned a non-null partial object also left that new object unowned.

The constructor now clears the old owner before checking destruction status,
retains a replacement output before checking creation status, and releases all
remaining objects on failure while the Workspace callbacks remain alive. Cleanup
errors do not replace the original installation error. Successful setup, sparse
solve arithmetic and capture operations remain unchanged.

| Regression | Result |
|---|---|
| Original fault-injection cases | 12 failed |
| Corrected negative cases plus successful ownership control | 13 passed |
| Pinned Warp1.18/FSAI downstream suite | 137 passed,87.39s |
| Stock Warp1.15 downstream suite | 137 passed,116.40s |

Faults cover data destruction, handler registration and replacement creation,
null and non-null partial outputs, subsequent cleanup errors, native invocation
of allocator callbacks during destruction, and repeated release. An intermediate
run had10passes/2failures because the test required Python None rather than
accepting an empty ctypes pointer; the final assertion checks absence of native
ownership. All logs are retained, including that test correction.

Downstream GPU checks cover first capture, changed RHS and starts, refinement,
generalized transforms/pencils, complex connections/shifts and damped modes.
They use the supported0.8.0.10 binary and e1d1be4 binding, L40/driver570.158.01,
serialized GPU access and four CPU threads pinned to cores8–11. This repairs
resource handling, not a performance gap or the separate QR-stage issue#198.

The native mock reproducer, original failure outputs and source fingerprints
are included. Related: [#197](https://github.com/alecjacobson/warpack/issues/197),
[earlier binding ownership fixes](../cudss_lifecycle/README.md).
