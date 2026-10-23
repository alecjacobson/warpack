# Actual loaded cuDSS provenance (#193)

Reusable direct Anvil, FEM application, skinning KKT and strict-pencil reports now
record the loaded runtime version, shared-library hash, binding-source hashes,
installed package version and relevant environment overrides after measured work.
The helper resolves the loaded symbol through the dynamic loader, so a bare SONAME
or an explicit binary override is not mistaken for the pip package's library.
It does not import or preload the optional backend, and reads no device numerical
values. Source/binary files must remain unchanged while each benchmark worker runs.

Application comparisons and Anvil table aggregation reject known cuDSS records
with missing or mismatched actual backend identities. Pure-Warp rows do not claim
to use a cuDSS library cached by a preceding method. Historical files are preserved;
missing old identities are not guessed or backfilled. Reproduce historical
summaries using their recorded benchmark revision.

Validation: 31 targeted CPU provenance/gate/diagnostic checks pass. These cover
actual-runtime versus package-version mismatch, changed override after caching,
SONAME resolution, absent optional backend, missing package metadata, failed
runtime queries, path-independent identities and mixed/missing comparison inputs.
The helper also records the verified 0.8 and 1.0 binaries in the separate
[strict-pencil qualification](../cudss_1_qualification/PENCIL.md).

Two sequential FEM methods pass with correct attribution: cuDSS identity for the
direct inverse, then no cuDSS dependency for pure-Warp CG despite the cached library.
Warp and CuPy skinning KKT adapters both pass their original checks and record the
same actual backend. These small smoke runs are correctness/identity evidence;
their incidental times are not performance comparisons.

The initial FEM smoke used a Hodge-only alternative Krylov adapter name for a
class-A problem and was rejected during setup. Its log and JSON are retained.
The corrected smoke selects the existing class-A `warp-cg` method with the same
problem, cycle budget and accuracy gates. Raw smoke records and launchers are here.
The downstream 137-test suite and 56-test stock compatibility suite also pass with
the [new pinned lifecycle fix](../cudss_lifecycle/README.md).
