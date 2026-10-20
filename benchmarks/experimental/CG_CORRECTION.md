# Fresh-residual correction for connection CG

Follow-up to [#95](https://github.com/alecjacobson/warpack/issues/95) and the
[connection comparison](../applications/CONNECTION.md). Residual correction is
now an optional public `CGInverse` setting:

```python
inverse = CGInverse(
    shifted_matrix,
    tol=1e-11,
    atol=0.0,
    preconditioner=preconditioner,
    refinement_steps=1,
    refinement_tol=0.01,
    verify_residuals=True,
)
```

The initial and correction solves both use unmodified upstream Warp CG and
share the same preconditioner. Each correction solves the fresh original
residual from a zero start and updates the solution in Warp. The correction
state uses relative `refinement_tol`, zero absolute tolerance and the same
maximum iteration budget. Optional final auditing uses the initial state's
effective tolerance against the original RHS, including Warp's dtype floor.
It never substitutes the much smaller correction RHS tolerance.

`stats` counts original RHS calls, total initial+correction iterations, and a
failed original RHS if any constituent native solve failed. `initial_stats`
and `refinement_stats` expose the native stages separately. `fresh_stats`
checks the final original equation. Stage arrays exist only when corrections
are enabled; the default correction budget is zero. Multiple RHSs, exposed
`b`/`x` in-place workspaces, and upstream BSR block shapes are supported.

The original one-RHS benchmark prototype is preserved in commit `aac7b5a`.
`corrected_cg.py` now only maps the recorded experiment's option names to the
public API. The data below use FSAI32 (FP32 factor / FP64 CG), initial relative
tolerance 1e-11 and correction tolerance 0.01 unless explicitly labeled otherwise.
All arithmetic and convergence checks execute in Warp.

## Evidence

L40 / driver 570.158.01 / pinned Warp 1.18, consistent complex mass, k=20, complex
ncv=64, σ=−0.1, three outer cycles, three warm synchronized samples. All original
physical accuracy/orthogonality/spectrum/subspace gates remain unchanged at 1e-8;
all 20 complex and 40 real physical convergence counts are required at 1e-10.

| Profile | Complex n | Fresh inner misses | Complex / real counts | Solve median (s) | Accepted |
|---|---:|---:|---:|---:|---|
| No correction, inner 1e-11 | 4096 | 2 / 298 | 10 / 20 | 0.8821 | No |
| One correction, inner 1e-11 | 4096 | 0 / 298 | 20 / 40 | 1.0133 | Yes |
| One correction, inner 1e-13 | 4096 | 6 / 298 | 20 / 40 | 1.0967 | No |
| One correction, inner 1e-11 | 16384 | 0 / 298 | 20 / 40 | 2.5857 | Yes |

The matched n=4096 control establishes that relaxing the inner request alone
does not restore strict stopping. Correction costs about 14.9% and restores all
counts and fresh checks. This is an accuracy improvement with measured cost,
not a speedup over the rejected control. The tighter 1e-13 experiment still
fails: its maximum fresh relative residual is 1.14e-12. Correction does not
guarantee an arbitrarily strict floating-point residual, and that old request
has not been relabeled as passing.

At n=16384, maximum original eigenpair residual is 1.65e-11, fresh relative linear
residual 4.54e-12, and all requested-space gates pass. Iterations are 50492 initial
+7331 correction = 57823 total. Setup 0.1825 s and capture/JIT/warmup 4.0387 s are
separate from replay. A matching regression replay gives 2.5904 s (ratio 1.001815),
passes all checks, and stays below the 1.25 performance gate. Relative to the
previous accepted cuDSS 0.3296 s, this still leaves about 7.8× replay cost, tracked
in [#97](https://github.com/alecjacobson/warpack/issues/97).

All profiles retain their raw samples and stopping evidence:
[matched control](../../results/connection-correction-control.json),
[n=4096 correction](../../results/connection-correction-screen.json),
[rejected tighter request](../../results/connection-correction-tight.json),
[n=16384](../../results/connection-correction-16384.json),
[baseline replay](../../results/connection-correction-regression.json).
Each has a corresponding `.occupancy.json` record. Accepted runs were
uncontended; failed runs remain rejected. Representative graphs contain no
HOST, HtoD or DtoH nodes.

The prototype checkpoint passed 50 tests on pinned Warp 1.18 in 36.61 s:
scaling, original-RHS preservation, in-place aliases, stage counters, zero and
nonfinite input/recovery, graph transfer audits, and repeated complex pencils
with changed starts. [Checkpoint validation](../../results/corrected-cg-prototype-validation.json).

```bash
python -m benchmarks.monitor_gpu --out build/correction.occupancy.json \
  --log build/correction.log -- python -m benchmarks.applications.connection_compare \
  --side 128 --modes 20 --ncv 64 --steps 3 --repeats 3 --cg-tol 1e-11 \
  --methods warp-cg-corrected --out build/correction.json
```

The pinned upstream Warp FSAI environment is required for this comparison.
`--correction-steps 0` is the explicit control; `--correction-tol` controls the
relative tolerance for the residual solves. Baseline gates include both values.

## Public implementation regression

The public implementation passes the saved n=16384 comparison at **2.5852 s**
(ratio **0.9998**), with the same 57,823 total iterations, all 20 complex / 40
real physical counts, and zero fresh inner misses. Its maximum fresh relative
linear residual is 4.54e-12. [Public replay](../../results/cg-refinement-public-16384.json)
and [occupancy record](../../results/cg-refinement-public-16384.occupancy.json).
The more demanding 1e-13 historical request remains failed; the accepted
configuration explicitly requests 1e-11 and adds correction. This is not a
claim that refinement can meet every requested floating-point tolerance.

An independent **host-only** roundoff diagnostic uses extended-precision
residuals/accumulation (64-bit significand on this machine) with six FP64
SuperLU correction solves. For a lowest-mode mass-image RHS, the reference
reaches relative residuals 7.3–8.2e-16 at n=4096 and 2.9–3.3e-15 at n=16384.
Simply rounding that reference vector to FP64 raises the residual, even when
evaluated in extended precision, to **7.5–7.7e-13** and **3.0–3.1e-12**.
FP64 residual evaluation adds further error. Random RHSs have much smaller
rounded residuals; scaling either RHS by 1e-6 or 1e6 barely changes the pattern.
The ideal Fourier/congruence condition bounds grow from approximately
15,484–61,138 to 61,937–244,552. This supports an RHS-dependent representation
and evaluation roundoff explanation for the strict failures. It is **not**
a proof of a minimum residual over every representable FP64 vector.
[Reference and all scales](../../results/performance_followup/cg-refinement-roundoff.json).

The full pinned suite passes **1,034 tests in 452.03 s**. The CG refinement,
audit, tolerance and baseline subset passes **57 tests on stock Warp 1.15 in
11.21 s**, including its explicit rejection of unsupported nested capture.
The public tests add multiple RHS scales, initial-versus-correction tolerance
separation, 2×2/3×3 BSR blocks, changed matrix values, and exposed workspace aliases.
[Validation](../../results/performance_followup/cg-refinement-validation.json).

With refinement disabled, the saved positive buckling regression at n41,616 /
k20 passes in **5.4729 s**, ratio **1.0013**, with exactly the previous graph
node/copy counts and no HOST/HtoD/DtoH nodes. This rechecks the positive branch;
the earlier negative-branch timing is not relabeled as rerun.
[Default regression](../../results/performance_followup/cg-refinement-default-regression.json),
[occupancy](../../results/performance_followup/cg-refinement-default-regression.occupancy.json).
