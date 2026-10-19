# Davidson-fixture and partial-SVD accuracy profiles, 2026-10-19

All four exported Spectra Davidson fixtures and all four partial-SVD fixtures
pass explicit wider-space profiles, at the original assertion bounds and
internal tolerance of **1e-12**. This resolves the accuracy acceptance criteria
of [#11](https://github.com/alecjacobson/warpack/issues/11) and
[#10](https://github.com/alecjacobson/warpack/issues/10) with documented settings.
**No production eigensolver code or default settings changed.** These are
untimed correctness runs, not speedup claims or full upstream conformance.

| Family | k | Old / validated ncv | Cycle limit | Observed cycles | Result |
|---|---:|---:|---:|---:|---|
| Davidson dense/sparse, LA/SA | 10 | 20 / 40 | 300 | 24–31 | 4/4 pass |
| SVD tall/wide, dense/sparse | 5 | 10 / 20 | 300 | 19–23 | 4/4 pass |

The Davidson source does not prescribe ncv: its exported zero means unspecified,
and Warpack's symmetric Krylov default chooses 20. Warpack does not implement
Davidson through this adapter. SVD uses an augmented symmetric formulation;
its ncv is not algorithmically equivalent to Spectra's SVD subspace size.

## Davidson diagnosis

The [fresh old-width control](../../results/davidson_before.json) reproduces the
two LA failures: absolute residuals **3.06e-10** and **2.87e-10**, against
**1e-10**. Both return 10/10 according to the correctly reported normalized
criterion `||Ax-lambda*x||_2 / max(1,abs(lambda)) <= 1e-12`. With eigenvalues near
1,000, this criterion does not imply the upstream absolute bound. The SA
controls pass. These direct solves use pure Warp, with no cuDSS or CG.

[Tightening tol to 1e-13](../../results/davidson_tighter_tolerance.json) at the
same width and budget does **not** improve the outputs. The
[diagnostic](../../results/davidson_diagnostic.json) shows both LA cases consume
all 300 cycles, without independent-chain recoveries. Cached operator products
agree with fresh host products to about 2.3e-12 absolute; there is no evidence of
an incorrect reported residual.

Repeated rotations accumulate vector norm errors around 1.5e-13. Recomputing a
normalized Rayleigh quotient on the host reduces eigenvalue errors from about
3e-10 to around 1e-12, but leaves residuals **1.33e-10 / 2.65e-10**: output
renormalization alone is insufficient. This host calculation is a diagnosis,
not a production correction. Narrow-space convergence and roundoff remain
relevant to [adaptive restart work #31](https://github.com/alecjacobson/warpack/issues/31).

With ncv=40, [all four cases pass](../../results/davidson_wider_basis.json).
Worst absolute residual is **9.06e-11**, spectrum error **9.06e-11**, subspace
error **1.12e-11**, and all counts reach 10/10. The runner now checks the requested
spectrum, orthogonality and invariant subspace, in addition to the original
absolute residual. [Comparison](../../results/davidson_profile_comparison.json).
The LA absolute error remains close to the 1e-10 bound; this profile is evidence
for these fixtures, not an accuracy guarantee for arbitrary scaled matrices.

## SVD convergence and stronger validation

The [old-width control](../../results/svd_before.json) reproduces the remaining
wide dense `SVD_1` failure: original numerical assertions pass, but the augmented
and physical counts are only 4/5 after 300 cycles. At ncv=20,
[all four cases pass](../../results/svd_wider_basis.json): the augmented count
and both left/right physical counts reach 5/5. Worst singular-value error is
**1.02e-12**, paired-equation residual **5.05e-13**, and subspace error
**1.38e-12**. [Comparison](../../results/svd_profile_comparison.json).

Validation now includes `A V = U Sigma` and `A^T U = V Sigma`, orthogonality,
and leading singular subspaces. The original componentwise absolute-vector
assertions can miss inconsistent left/right signs. A dedicated negative test
checks that such an invalid triplet is rejected. Strict success also requires
both physical convergence counts when recorded; an augmented count alone is
insufficient. Historical reports without these extra checks remain explicitly
limited to the evidence they recorded.

## Reproduction and validation

Pinned Spectra revision: `db1d5cc3279752ca7ea3e33da44ba2a85e4e4a95`.
The exporter executes its original generators and preserves original case IDs,
matrices, selections and k. Source SHA256 values:

- `test/DavidsonSymEigs.cpp`: `a10014938de28519087ab9a5b7c5c4995a5f52e409bb827ce405e58f9c8c8a5c`
- `test/SVD.cpp`: `53fe3cd06b81050a678830164d7a68e10cd54f7b680a94255c3f49eb6327860a`

```bash
python -m benchmarks.upstream.prepare_spectra --families DavidsonSymEigs SVD \
  --max-size 1000 --out build/accuracy-profiles/fixtures
export PYTHONPATH="$PWD/build/warp-fsai:$PWD"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python -m benchmarks.upstream.run_spectra --fixtures build/accuracy-profiles/fixtures \
  --family DavidsonSymEigs --accuracy-only --ncv 40 --out results/davidson_wider_basis.json
python -m benchmarks.upstream.run_spectra --fixtures build/accuracy-profiles/fixtures \
  --family SVD --accuracy-only --ncv 20 --out results/svd_wider_basis.json
python -m benchmarks.upstream.davidson_accuracy_audit \
  --fixtures build/accuracy-profiles/fixtures --out results/davidson_diagnostic.json
```

Omit `--ncv` to reproduce the old-width profiles. Add `--tol 1e-13` to the
old-width Davidson run for the rejected tolerance-only control. Neither override
changes the original numerical assertion bounds. The runner records effective
ncv, requested internal tolerance and explicit overrides. Accuracy runs capture
and replay twice, with host reference validation outside the graph.

Validation on NVIDIA L40, Warp 1.18.0+cu12 with the pinned FSAI overlay:

- **69 tests pass** across symmetric accuracy, SVD, upstream reporting contracts
  and the real symmetric/generalized solver tests.
- **46 tests pass on stock Warp 1.15**, covering symmetric accuracy, SVD and
  reporting contracts.
- Native pinned Spectra passes **12 Davidson assertions** and **16 SVD assertions**.
- New symmetric regressions check captured replay and graph structure: no host
  callbacks or HtoD/DtoH copies. The exact previously failing SVD wide fixture
  and its transpose check both equations, subspaces and strict counts on replay.

The new compressed SVD fixture is about 744 KiB. No full 1000-by-1000 Davidson
binary is added; generated small stress tests and the reproducible upstream
harness provide its coverage. Large Anvil/POD timing tables remain unchanged.
