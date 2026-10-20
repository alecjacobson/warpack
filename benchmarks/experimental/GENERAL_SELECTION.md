# General spectrum-selection diagnostics

The public general solver's original Spectra benchmark profile passes 11/20
requested-set checks. Doubling the explicitly requested `ncv` improves this to
19/20 across three reset graph replays per case. This is an accuracy profile,
not a solver fix or a performance result. The original fixtures, starts, k=10,
tol=1e-10, and maximum 300 device-controlled restart cycles are unchanged.
Widths become 40 for n=100 and 60 for n=1,000. The public retained width remains
14. Production defaults and historical timings are unchanged.

[Full wide-profile results](../../results/general-selection-wide.json) include
all 60 replay validations, values, source/runtime fingerprints, input hashes,
device counts, and executed cycles. Accepted replays check original residuals,
complete target assignment, required eigenvalues above boundary ties, normalized
independent vectors, and the allowed invariant subspace. These diagnostics do
not time execution.

`benchmark_gen_38` remains rejected in all three wide-profile replays: each
stops after 13 cycles with count 10 and a normalized residual below 1e-10, but
returns the wrong requested LM set. The eigenvalue assignment error is about
31.47. The [native reference audit](../upstream/GENERAL_SELECTION.md) also rejects
real Spectra and ARPACK for this case at width 60; complex ARPACK passes. All
native variants pass with the separately labeled width-120 profile. This does
not establish that an arbitrary problem's requested set can be certified from
local residuals.

The source change extracts the existing `sort_complex` launch into `select()`
so diagnostic subclasses can inspect or alter ordering without intercepting
Warp's global launch function. The default selection arithmetic is unchanged.
Exploratory guard and real-basis restart variants are not promoted: recovering
some target sets did not reliably meet both spectrum and residual gates. The
public algorithm, independent-reference coverage, and performance follow-up
remain tracked in [#104](https://github.com/alecjacobson/warpack/issues/104) and
[#14](https://github.com/alecjacobson/warpack/issues/14).

Reproduce using the pinned Warp environment and exported original fixtures:

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  PYTHONPATH="$PWD/build/warp-fsai:$PWD" \
  python -m benchmarks.experimental.general_selection_profiles \
  --ncv-multiplier 2 --budget 300 --repeats 3 \
  --out build/general-selection-wide.json
```

Use `--fixtures` for an existing export and repeated `--case` to select a
subset. Capture warmup precedes the reset replays; original v0 is read again
inside every replay. No additional vendor eigensolver runs inside Warp.
