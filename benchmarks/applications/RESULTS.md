# Application coverage results

This is the initial application baseline. The [cuDSS correction report](../upstream/REFINEMENT.md)
records the subsequent damped-chain fix, current tests, and interior-mode timing regression.

Implementation: `adf397f`, NVIDIA L40, Warp 1.18.0+cu12, cuDSS 0.8.0.10. See the JSON files for exact library versions, matrices, samples, setup and capture times. Each solve is warmed once and measured three times. Initial contended trials were excluded from these reports.

Local verification: **128 passed, 1 strict expected failure**, plus lint, formatting and wheel-build checks. The expected failure is the existing nonsymmetric solver on the new damped-chain fixture, tracked in [#27](https://github.com/alecjacobson/warpack/issues/27).

## Coverage and remaining failures

At requested sizes 80 and 1024, there are **72 matrix/method configurations: 57 numerical passes, 3 numerical failures and 12 unsupported adapter paths**. The failures are Warp damped dynamics at both sizes, and CuPy LOBPCG on the larger conformal pencil. Unsupported paths are documented in the [protocol](README.md); they are not passes.

Warp passes the numerical checks for A–G and I. All cuDSS-backed Warp rows fail the stricter zero-host-transfer graph check because the vendor solve captures two scalar HtoD copies per application; see [#29](https://github.com/alecjacobson/warpack/issues/29). They remain graph capturable, with no host callback or DtoH node observed. SVD and the three tested pure-Warp CG graphs pass the device-only graph check. No CUDA API trace was performed.

The initial CuPy LOBPCG tolerance of `1e-10` caused over-iteration and failed results on some FEM bars. Using absolute tolerance `1e-8` passes the unchanged independent accuracy gate. The larger conformal case remains unreliable. CuPy connection-ring `eigsh` needs ncv=64 at n=1024 in this run; ncv=32 exhausted its iteration budget with residual 2.60e-7. Previous attempts are retained inside the final records. These observations do not establish universal library defects.

## Larger stiffness/mass comparisons

10,000-DOF P1 bars with non-diagonal consistent mass; median solve milliseconds, excluding factorization, capture and output conversion. All timings below pass the same original-equation, spectrum and mass-orthogonality gates. CuPy here is **generalized LOBPCG + cuDSS**, not eigsh. Warp uses five generalized Krylov cycles.

| Target | ARPACK | Spectra | CuPy LOBPCG + cuDSS | Warp + cuDSS |
|---|---:|---:|---:|---:|
| Lowest modes | 11.73 | 19.83 | 72.14 | 20.48 |
| Interior modes | 11.02 | 19.86 | unsupported | 21.13 |
| Most-negative modes | 10.57 | 19.79 | 67.89 | 20.50 |

These banded 1D systems favor the CPU references and do not establish 3D FEM performance. They exercise large consistent-mass pencils with independently known spectra. The next replay changed Warp timings by less than 0.2%, with all accuracy gates still passing; the regression threshold was 25%.

## Generalized convergence improvement

The exact Spectra `SymGEigsShift_5` fixture (n=100, k=10, ncv=20, sigma=1.2345) is now a permanent test. The new opt-in `GeneralizedKrylovSchur` path uses the mass inner product and evaluates physical Rayleigh quotients and residuals. The old block API is unchanged.

| Method | Cycles | Median | Maximum absolute residual | Converged |
|---|---:|---:|---:|---:|
| Existing block method | 100 | 37.27 ms | 4.85e-6 | 5/10 |
| Existing block method | 300 | 110.60 ms | 7.33e-15 | 10/10 |
| New generalized Krylov | 20 | 17.93 ms | 8.60e-15 | 10/10 |

The matched-accuracy improvement is **6.17× on this small regression**. Requested eigenvalues, B-orthogonality and invariant-subspace agreement are tested independently. This is not a large-mesh speedup claim, and [#7](https://github.com/alecjacobson/warpack/issues/7) remains open for the other generalized cases, including buckling and permitted nonconvergence.

## Existing Anvil regression checks

The same 283,857-DOF Anvil problem, starts and solver settings pass their existing physical residual, eigenvalue, modal-correlation and rigid-subspace checks.

| Existing path | Published baseline | Current median | Change |
|---|---:|---:|---:|
| Warp + cuDSS | 0.466910 s | 0.466930 s | +0.004% |
| Warp + FSAI64f-CG | 9.608229 s | 9.586478 s | −0.226% |

The CG baseline enabled profiling; the current run disables profiling. These measurements show no material regression, not an optimization of either existing Anvil path. The approximately 20× CG/direct gap remains [#15](https://github.com/alecjacobson/warpack/issues/15).

## Evidence

* [All nine classes, two resolutions](../../results/application_coverage.json)
* [10,000-DOF comparison](../../results/application_coverage_large.json)
* [cuDSS / block / pure-Warp CG comparison](../../results/application_inverse_comparison.json)
* [Repeated large-problem regression check](../../results/application_performance_regression.json)
* [Exact Spectra regression](../../results/generalized_krylov_regression.json)
* [Anvil cuDSS](../../results/anvil_application_regression.json) and [FSAI-CG](../../results/anvil_cg_application_regression.json)
* [Validation, provenance and isolated cuDSS graph audit](../../results/application_validation.json)

These fixtures establish initial coverage, not completion of the broader application contracts. See [#20–#28](https://github.com/alecjacobson/warpack/issues/18) for clustered spectra, full nullspace completeness, constraints, richer mass matrices, complex/generalized APIs, and larger application-specific workloads still needed.
