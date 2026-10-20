# Hodge strict-count budget follow-up

[#103](https://github.com/alecjacobson/warpack/issues/103) records a full-suite
failure in the existing direct generalized Krylov topology regression:
5/6 strict counts at n294, k6/ncv32, tolerance 1e-11 and three fixed cycles.
Original residual, four-dimensional harmonic space and positive guard checks
passed. The real generalized path does not use complex projected Schur.
[Failed-run record](../../results/performance_followup/hodge-count-full-failure.json).

The failure has not reproduced in the isolated three-cycle controls: 30 replays
with one setup and 36 replays across 12 fresh setups pass. Four-cycle controls
also pass, with maximum reported residual below 7.3e-13. These observations do
not erase the full-run failure or establish its detailed numerical cause.
[Isolated controls](../../results/hodge-count-isolated.json),
[fresh-setup controls](../../results/hodge-count-setups.json).

The topology/accuracy regression now reserves **six fixed direct cycles**.
It does not change the solver, CG profiles, tolerances, strict count checks,
physical checks or published performance tables. This is an explicit larger
test budget, not a claim that the three-cycle profile is reliable or fast.
36 replays across 12 new setups pass at six cycles, and all 26 focused Hodge
tests pass. Broader suite validation remains required before closing #103.
[Six-cycle evidence](../../results/hodge-count-reserve.json).

```bash
python -m benchmarks.experimental.hodge_count_probe --budgets 3 6 \
  --setups 12 --repeats 3 --out build/hodge-counts.json
python -m pytest -q tests/test_hodge.py
```
