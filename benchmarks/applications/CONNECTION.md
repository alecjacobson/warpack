# Complex connection pencils and positive mass

This expands [class E / #24](https://github.com/alecjacobson/warpack/issues/24)
from a standard connection ring to complex Hermitian stiffness / SPD mass
pencils. Smooth direction fields motivate this class; see Knöppel et al.,
[Globally Optimal Direction Fields](https://www.cs.cmu.edu/~kmcrane/Projects/GloballyOptimalDirectionFields/).
The fixture is a manufactured **flat periodic P1 triangular U(1) connection**,
not a reproduction of that paper's curved-surface discretization.

`connection.py` builds a side × side periodic grid with holonomies (0.37,0.91),
vertex phase gauges, and nonuniform amplitude congruence. Mass can be identity,
positive diagonal lumped, or consistent with complex off-diagonal entries.
Disconnected copies have exactly repeated complex spectra. Independent Fourier
symbols supply the requested eigenvalues and full boundary eigenspaces; a
separate triangle-gradient/local-mass calculation checks the assembled energy.
Dense generalized LAPACK checks the small fixtures. Reference vectors are
never supplied as iteration starts.

## API boundaries and comparisons

The installed library's complex Hermitian API remains **standard-problem**
support. `connection_warp.py` is an explicitly separate application adapter:
it composes public `HermitianCSR` realification and `GeneralizedKrylovSchur`,
then removes duplicate complex directions with two-pass **mass-weighted**
Gram–Schmidt in Warp. Physical n/k/ncv become real 2n/2k/2ncv. Recovered complex
pairs receive fresh residual checks against the original K and M; zero-filled
missing rows cannot count as converged. Changed cuDSS coefficients require new
factors and graphs. No NumPy/SciPy/CuPy computation occurs in its replay.

* Warp+cuDSS uses realified general factors, matching, and one FP64 residual
  correction through `ComplexShiftInverse`. All other arithmetic is Warp.
* Pure Warp uses upstream FSAI32 (FP32 factors) and FP64 `CGInverse`, or upstream
  diagonal CG in tests. Fresh inner residual auditing is enabled. Large strict
  inner failures remain tracked in [#95](https://github.com/alecjacobson/warpack/issues/95).
* SciPy uses complex ARPACK with a supplied SuperLU inverse; setup excludes
  solve timing. Its complex `eigsh` dispatches to the nonsymmetric ARPACK engine.
  `scipy-ritz` additionally performs a timed small generalized Ritz solve in
  the returned space, making repeated modes M-orthogonal. It cannot manufacture
  missing eigendirections. The raw unmodified result is retained separately.
* Spectra uses its pinned **native complex `HermEigsBase` engine**, custom
  inverse-M and M-metric adapters, and Eigen complex `SimplicialLDLT` of K−σM.
  There is no additional M factorization and no claim of a turnkey public
  `HermGEigsSolver` interface. Pinned Spectra commit:
  `db1d5cc3279752ca7ea3e33da44ba2a85e4e4a95`.
* Native complex CuPy `eigsh` covers identity/positive diagonal mass by explicit
  congruence; it has no coupled generalized mass argument. Coupled mass uses
  native complex LOBPCG with B=M and a columnwise realified cuDSS preconditioner.
  CuPy 14.2.0 has a confirmed complex whitening bug, described below.
  **`cupy-corrected` changes two helpers only in its isolated process. It is a
  diagnostic modified-CuPy reference, not the installed native result.**

## Measured results

NVIDIA L40, driver 570.158.01, Warp 1.18 pinned FSAI overlay, cuDSS 0.8.0.10,
CuPy 14.2.0, SciPy 1.15, four OpenBLAS/OMP threads. Every accepted row passes
original residual, spectrum, requested-space and conjugate metric-orthogonality
gates 1e-8. Warp additionally requires all 20 complex and 40 real physical counts
at 1e-10. CuPy LOBPCG requires fresh and reported absolute residuals ≤1e-9.
Setup, capture/JIT, and one untimed warm solve are separate; solve is the median
of three synchronized wall samples. Every timed sample is independently checked.

For **16,384 complex unknowns, k=20, complex ncv=64, consistent mass, σ=−0.1**:

| Method | Setup (s) | Capture (s) | Solve (s) | Max normalized original residual, first trial |
|---|---:|---:|---:|---:|
| SciPy ARPACK + complex SuperLU | 0.2417 | — | 0.6201 | 5.93e-13 |
| Spectra complex engine + Eigen LDLT | 0.2554 | — | 0.7464 | 7.45e-13 |
| **Modified** CuPy LOBPCG + cuDSS | 0.4173 | — | 0.4251 | 1.20e-10 |
| Warp generalized adapter + cuDSS | 0.3838 | 0.9698 | **0.3296** | 5.90e-13 |

Warp uses three fixed cycles, real dimension 32,768 and real ncv=128. It is about
1.9× faster than SciPy and 2.3× faster than Spectra on this fixture; these are
solve comparisons, not first-use totals. A separate accepted baseline replay
gives 0.3291s Warp (ratio 0.9984) and 0.4220s modified CuPy (ratio 0.9927).
The saved gate requires matching matrix/configuration/runtime hashes, accepted
aggregates, uncontended rows and a solve ratio ≤1.25.
[Main record](../../results/connection-16384.json),
[occupancy](../../results/connection-16384.occupancy.json),
[baseline regression](../../results/connection-16384-regression.json).

At n=4096 consistent mass, three cycles give 0.2011s Warp+cuDSS versus 0.1598s
SciPy and 0.1669s Spectra. Eight cycles took 0.4949s; reducing the explicit budget
preserves all gates. Those aggregate experiments also requested CG and are
**rejected aggregates** because the CG stopping contract fails; their accepted
individual direct rows are observations, not saved accepted baselines.
The n=4096 identity-mass comparison passes native CuPy eigsh at 0.2040s and the
Warp generalized adapter at 0.1898s. Public standard `ShiftInvertEigensolver`
is separately regression-tested on the same connection family. At n=1024 with
positive lumped mass, native CuPy eigsh with diagonal congruence passes at
0.0589 s versus 0.1457 s for the Warp generalized adapter; this smaller case
does not favor realified shift-invert.

## Failures retained and isolated

**CuPy complex whitening** ([#94](https://github.com/alecjacobson/warpack/issues/94),
[upstream #10405](https://github.com/cupy/cupy/issues/10405)):
`_b_orthonormalize` and generalized `_eigh` use `inv(L.T)` instead of
`inv(L.T.conj())`. A native-only 12×12 reproducer gives orthogonalization error
2.18 and generalized residual 7.47. Correcting only those expressions gives
8.89e-16 and 1.78e-15. Native-only n=144 connection solves without a preconditioner
return residuals 0.997/0.984/0.943 for identity/lumped/consistent mass; the
two-helper correction gives 1.78e-11/9.28e-11/7.90e-11. The original
cuDSS-preconditioned call throws `eigh has failed in lobpcg iterations`.
The native-only diagnostics import neither Warp nor cuDSS. Installed CuPy remains unchanged.
[Original](../../results/connection-cupy-native.json) and
[corrected diagnostic](../../results/connection-cupy-corrected.json).

**Repeated ARPACK basis** ([#96](https://github.com/alecjacobson/warpack/issues/96)):
two copies, n=2048,k=20,consistent mass have the correct raw ARPACK spectrum and
subspace but mass-orthogonality error 0.3514. The explicitly labeled Ritz adapter
reduces that error to 2.0e-15 at 0.1137s; raw 0.1057s is rejected. Spectra, modified
CuPy and Warp pass the same repeated-space checks.
[Original](../../results/connection-repeated.json),
[timed recovery](../../results/connection-repeated-ritz.json).

**CG fresh residuals**: n=4096,8 cycles,inner tol 1e-13 gives correct modes
(original residual 1.32e-11, full physical counts), but 59/723 fresh linear
residuals miss the requested tolerance despite zero recursive misses; maximum
fresh relative residual 7.74e-12. Its 2.375s timing is diagnostic. Using three
cycles and tol 1e-11 reduces runtime to 0.882s but still misses two fresh inner
checks and half the strict outer counts. Both original-equation 1e-8 checks pass;
neither configuration is accepted as a strict solve. Smaller diagonal-CG
regressions pass with tol 1e-12. Do not infer arbitrary-precision inner accuracy
from recursive convergence or hide these failures by relabeling tolerances.
[Eight-cycle evidence](../../results/connection-large-initial.json),
[three-cycle evidence](../../results/connection-large-budget3.json).

The large cuDSS graph has no HOST callbacks or DtoH transfers, but 1192 vendor
HtoD metadata copies (two factor solves per apply, including correction): the
existing [#29](https://github.com/alecjacobson/warpack/issues/29) limitation.
Pure-CG graphs have no HOST/HtoD/DtoH nodes. These are DOT audits, not API traces.

## Reproduction and scope

```bash
g++ -O3 -DNDEBUG -std=c++17 -shared -fPIC -I/usr/include/eigen3 \
  -Ibuild/upstream-spectra/include benchmarks/applications/connection_spectra.cpp \
  -o build/connection_spectra.so
python -m benchmarks.monitor_gpu --out build/connection.occupancy.json \
  --log build/connection.log -- python -m benchmarks.applications.connection_compare \
  --side 128 --modes 20 --ncv 64 --steps 3 --repeats 3 \
  --methods scipy,spectra,cupy-corrected,warp-cudss --out build/connection.json
python -m benchmarks.applications.connection_cupy_repro --out build/cupy-native.json
python -m benchmarks.applications.connection_cupy_repro --correct-helpers \
  --out build/cupy-corrected.json
```

Set the pinned Warp overlay and four-thread environment from the main README.
Use `--baseline` with a matching accepted report; native failed methods cause
nonzero process exits while retaining all evidence. Small tests cover triangle
energies, gauge changes, multiplicity, boundary eigenspaces, changed starts,
short budgets, missing rows, nonfinite input/recovery, and graph transfer audits.
This is application-adapter coverage; it does not add a public complex
generalized convenience API, certify arbitrary curved connection discretizations,
or close the remaining CG and upstream CuPy issues.

All 39 new tests (25 host, 14 GPU) pass on pinned Warp 1.18 in 30.47 s and stock
Warp 1.15 in 31.17 s. [Validation record](../../results/connection-validation.json).
No installed-core files changed; the earlier full 941-test result is retained
as that earlier run, not relabeled as a fresh full-suite run.
