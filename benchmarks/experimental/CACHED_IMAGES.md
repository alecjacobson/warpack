# Reusing the current block's operator image

`CachedXBlockLOBPCG` is an experimental subclass outside the installed package.
It carries the freshly computed `A X` through the same invariant projections,
row scaling and two CholeskyQR transformations as `X`. The Ritz step then applies
`A` only to the new and history search directions. Final physical residuals
still recompute `A X` from the original operator on every iteration. Reuse does
not accumulate operator images over multiple iterations.

For excluded invariant vectors `Q`, the adapter also carries `A Q` through
projection. This supports nonzero invariant eigenvalues, not just rigid modes
where `A Q = 0`. Initialization refreshes that image after the current constraint
basis is refreshed and normalized, including on graph replay. Matrix topology
and buffer addresses remain fixed. Preconditioner factors remain caller-managed.

The prototype hooks the public solver's private launch/projection methods;
it is tied to this checkout and is **not a supported subclassing API**. It
reuses existing scratch arrays and allocates only the additional `A Q` image.
There are no new numerical host decisions. The public solver and defaults
remain unchanged.

Twenty regression cases pass in the pinned Warp environment (**3.09 s**) and
on stock Warp 1.15.0 (part of a 45-test focused run, **8.34 s**).
[Validation and hashes](../../results/performance_followup/cached-images-validation.json).
These cover ordinary and coupled SPD mass, SPD/indefinite
stiffness, repeated modes, rigid and nonzero invariant exclusions, both tuning
configurations, and changed matrix/mass/constraint values on the same graph.
They independently check every carried `A S` image against the current original
matrix, as well as the requested eigenvalues/subspace, physical residuals,
orthogonality, counts, readonly starts and graph transfers. Changed starts come
from computed modes, with explicit exploration across crossings.

## Measurements

All GPU jobs ran sequentially on the L40. Adjacent occupancy reports observed
no competing processes; clocks were not locked. Every recorded trial passes
shared physical, requested-spectrum/subspace, orthogonality and status/count
gates. CuPy also passes its own freshly evaluated stopping tolerance.

| Case | Cached images Warp | CuPy, same preconditioning |
|---|---:|---:|
| Anvil, 283,857 DOFs, 20 elastic modes | 3.3060 s | 3.1409 s |
| Disconnected Hodge, 39,366 DOFs, four harmonics + two guards | 72.42 ms | 118.55 ms |

The fresh Anvil control without image reuse is **3.3739 s**, versus **3.2153 s**
for CuPy. The preceding active-CG report was **3.3398 / 3.1462 s**. Image reuse
therefore gives a small measured gain, approximately 1–2% depending on control,
and does not establish parity with CuPy. Warp takes 28–29 outer steps; small
floating-point changes also alter the active CG counts and convergence path.
This is not an isolated sparse-product throughput comparison.

The cached Anvil run's worst returned-pair residual is **9.6971e-9**, and its
worst original residual **1.9395e-8**. Setup is **1.2190 s**, capture **.6022 s**,
and setup/capture device free-memory decrease **4.308 GB**, including pools and
runtime allocations rather than a peak-memory measurement. Its graph contains
13 conditional nodes, 523 kernels and 38 device copies, with no HOST callback,
HtoD or DtoH transfer in the DOT audit. Only the selected CG bucket executes.

The 10,890-DOF related consistent-mass FEM case shows essentially no gain:

| Start | Cached images Warp (s) | Previous active-CG Warp (s) |
|---|---:|---:|
| Random | .29240 | .29045 |
| Previous 20 modes + random guards | .22115 | .22116 |
| Previous full 24-row block | .17320 | .17262 |
| Previous full block + explicit 1e-3 exploration | .18774 | .18691 |

All trials and matched unperturbed CuPy stopping checks pass. The exploration
row is a separate Warp policy measurement. Reference modes never enter starts.
These small differences do not justify promoting the optimization into the
core API. The experiment remains available for further profiling; #67 and
the larger iterative-versus-direct gap in #15 remain open.

Raw evidence: [Anvil trials](../../results/performance_followup/anvil-cached-x.json),
[fresh control](../../results/performance_followup/anvil-cached-x-control.json),
[screen/profile](../../results/performance_followup/anvil-cached-x-screen.json),
[Hodge](../../results/performance_followup/hodge-cached-x.json),
[related FEM](../../results/performance_followup/related-consistent-cached-x.json).
Use the commands in [ACTIVE_CG.md](ACTIVE_CG.md) with `--reuse-x-image` to reproduce.
