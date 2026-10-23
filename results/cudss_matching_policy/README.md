# Explicit matching policy across cuDSS versions (#196)

`CuDSSInverse(..., matching=False)` now sets native matching to NONE before
analysis. Previously it left the option unset: a native config probe returns
NONE (0) on the supported 0.8 runtime but AUTO (6) on the isolated 1.0 candidate.
`matching=True` still requests maximum diagonal product (5), with the same
matrix/reordering restrictions and Warp refinement budget. The default 0.8
numerical policy and dependency pin are unchanged. No replay operation changes.

| Check | Result |
|---|---|
| Old policy, CPU configuration/error tests | 6 failed, 2 passed |
| Explicit policy, same CPU tests | 8 passed |
| Supported cuDSS0.8 downstream checks | 68 passed,23.99s |
| Isolated cuDSS1.0 policy/replay checks | 15 passed,0.99s |

Real GPU tests read back the native configuration and compare each of three
changed RHS replays against its original matrix and a dense reference. SPD and
mixed-sign general factors cover widths1/3 and both allowed matching settings;
relative original residuals must remain below1e-12 and RHS values unchanged.
The symmetric-indefinite control checks explicit NONE and eager accuracy without
changing AUTO pivoting or claiming its cuDSS1.0 capture path is supported.

The test binding remains e1d1be41decb8d6a4305acdd40e26a5a86c0f797. GPU checks used
L40/driver570.158.01, stock Warp1.15, four CPU threads pinned to cores8–11,
and serialized GPU access. The 1.0 binary was selected in a separate process;
the globally installed 0.8.0.10 wheel was not replaced. The initial probe needs
no GPU arrays or solve. Logs, source hashes and probe outputs are retained here.
No performance improvement or full cuDSS1.0 qualification is claimed.

Related: [#196](https://github.com/alecjacobson/warpack/issues/196),
[cuDSS1.0 qualification](../cudss_1_qualification/README.md).
