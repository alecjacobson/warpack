# Fixed Arnoldi-budget controls for strict pencils (#123)

Increasing the fixed cycle budget is not a generic recovery for the strict original-residual floor. The gyroscopic 8- and 16-cycle profiles pass all four outputs at the original 1e-9 native tolerance; every damped/nonproportional profile still rejects at least one output. All 32 first/later outputs remain archived, including 17 native failures. All 32 pass the separate physical-quality gates. No solver code/default or acceptance threshold changes.

| Kind | Cycles | Counts: first, then three replays | Worst native residual | Complete worker accepted |
|---|---:|---|---:|---|
| [nonproportional](nonproportional-2-a1.json) | 2 | 16 / 19 / 16 / 16 | 1.649e-09 | False |
| [nonproportional](nonproportional-4-a1.json) | 4 | 18 / 17 / 20 / 20 | 1.974e-09 | False |
| [nonproportional](nonproportional-8-a1.json) | 8 | 19 / 20 / 20 / 19 | 1.332e-09 | False |
| [nonproportional](nonproportional-16-a1.json) | 16 | 18 / 20 / 20 / 19 | 2.842e-09 | False |
| [gyroscopic](gyroscopic-2-a1.json) | 2 | 16 / 17 / 18 / 17 | 1.882e-09 | False |
| [gyroscopic](gyroscopic-4-a1.json) | 4 | 20 / 19 / 19 / 19 | 1.711e-09 | False |
| [gyroscopic](gyroscopic-8-a1.json) | 8 | 20 / 20 / 20 / 20 | 9.647e-10 | True |
| [gyroscopic](gyroscopic-16-a1.json) | 16 | 20 / 20 / 20 / 20 | 9.838e-10 | True |

Each worker uses the same 65,536-unknown assembled pencil, 20 modes, ncv 64, shift -0.15+6.3i and one cuDSS residual correction per inverse. It changes only the explicit 2/4/8/16 fixed-cycle budget. The existing device checkpoint accepts complete sets using original-pencil residuals; no independent-vector polishing, quotient update or application-specific reconstruction is introduced. The 8-cycle gyroscopic observation is a separate profile, not a replacement for the rejected 2/4-cycle profiles. Finite samples do not establish universal convergence.

Every source/input/dependency identity matches within each pencil family. All occupancy samples are clean; nonzero worker exits and timing-ineligible records preserve numerical rejection. These are accuracy diagnostics, not speed comparisons. Each report retains original/native, extended-precision and stored-coefficient-reference checks, with no HOST nodes in the captured graph. Small vendor cuDSS metadata transfers remain within the existing backend limitation.

The direct-pencil numerical core is unchanged from the earlier [correction/filter controls](../README.md); these runs use repository source through `d5f5b1e` plus documentation-only updates. The computed-metric QR optimization affects the symmetric block solver and is not used here. Exact commands, arguments and hashes are retained in the adjacent JSON/log/occupancy records. #123 stays open for a generic strict-accuracy remedy.
