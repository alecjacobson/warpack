# Strict-pencil control: cuDSS 0.8 versus 1.0

The candidate does not resolve #123. Both versions use exactly the same assembled matrices, start, analytic and stored-coefficient references, 20 requested modes, ncv 64, one Warp residual correction, and unchanged native tolerance 1e-9. There are 16,384 physical nodes; realification yields the 65,536-dimensional sparse solve. Two and eight outer cycles were predeclared for each damping family and binary. This is an accuracy diagnostic; no timing claims are made.

All 32 outputs pass the existing physical gates. Native counts and strict acceptance still vary near the stopping floor:

| Family | Cycles | Runtime | Counts, first plus three replays | Strict passes | Largest native residual |
|---|---:|---|---|---:|---:|
| gyroscopic | 2 | 0.8.0 | [16, 18, 17, 16] | 0/4 | 2.763e-09 |
| gyroscopic | 2 | 1.0.0 | [17, 18, 16, 17] | 0/4 | 2.352e-09 |
| gyroscopic | 8 | 0.8.0 | [20, 20, 20, 20] | 4/4 | 8.674e-10 |
| gyroscopic | 8 | 1.0.0 | [20, 20, 20, 20] | 4/4 | 9.877e-10 |
| nonproportional | 2 | 0.8.0 | [19, 18, 18, 19] | 0/4 | 1.936e-09 |
| nonproportional | 2 | 1.0.0 | [17, 18, 18, 19] | 0/4 | 1.294e-09 |
| nonproportional | 8 | 0.8.0 | [20, 20, 20, 20] | 4/4 | 9.904e-10 |
| nonproportional | 8 | 1.0.0 | [20, 17, 18, 20] | 2/4 | 1.874e-09 |

The fresh 0.8 nonproportional eight-cycle control passes all four outputs, unlike an earlier independent run of that control. Its largest residual is only slightly below 1e-9; this fresh success does not erase previous rejected records or establish robust convergence. The 1.0 eight-cycle nonproportional control still rejects outputs, while both versions pass the eight-cycle gyroscopic control. No gates or iteration defaults were changed to turn these results into passes.

Every record includes actual loaded binary/runtime and binding source identities. Source, matrix/reference/input and dependency hashes are checked; sampled GPU occupancy is clean. The monitor marks workers with a nonzero numerical-rejection exit code as timing-ineligible; those records are retained as accuracy evidence, not rerun or relabeled successful.

[Qualified records](pencil/qualified.json), [earlier cycle-budget controls](../pencil_refinement/cycles/README.md). Raw JSON/logs and exact launchers are in `pencil/`.
