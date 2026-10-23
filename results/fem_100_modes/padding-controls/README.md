# Initial CuPy width control (#147)

`cupy100-a1` keeps the same 104-column Warp start/configuration as the post-#154
wide control, but gives CuPy its first 100 columns. It requests the same 100
physical modes, removing CuPy's four extra starting/padding modes. Both methods
pass all three measured replays and native criteria. CuPy history lengths are
170, 171 and 172, versus substantial variability in the earlier width-104
profiles. Its worst original normalized residual is 2.45e-11; Warp's is
6.01e-12.

The entire worker was contended, so **no timing comparison is qualified**.
This is evidence that width is worth studying, not proof that padding alone
explains the earlier timing gap. The tighter CuPy tolerance control remains
pending. Exact procedural matrix/reference/start fingerprints are also pending
in #156; configured inputs and transitive source/dependency provenance are
recorded. The CuPy input is a prefix of the same Warp start within this worker.

The current generator changes the first columns when Warp width changes, even
with the same seed, because it generates a row-major `(n,width)` array. Future
Warp-width comparisons must use a shared generated pool and matching prefixes;
do not call those starts identical based only on the RNG seed.
