# Source-to-wheel identity audit (#181)

A direct wheel build from a long-lived checkout included retired `warpack/fem.py`
and `warpack/skinning.py` from stale `build/lib`. Neither file was present in
current source. The exact inventory check rejected that local artifact before
numerical smoke tests or publication. This is not evidence of contamination in
published releases: the existing documented/CI build goes through a source
distribution, and preceding release checks used clean source staging.

The reusable [audit](../../scripts/audit_release_wheel.py) now checks:

- Exact current core Python module inventory, rejecting extra/missing modules and
  application Python files outside the core package.
- Byte equality to current source, plus optional exact installed-tree equality.
- Duplicate archive names and malformed zip archives.
- Rejection reports with artifact identity, and refusal to overwrite evidence.

All **11 CPU regression tests pass** ([transcript](tests.log)). They exercise
stale/missing/modified modules, unexpected application code, duplicate names,
corrupt archives and a stale installation. The tool requires no GPU or optional
numerical package.

The [rejected direct build](rejected-direct-build.json) has SHA256
`7b4d5a9c023da9c3c643ea7cdda721905dcd43d92fada7be4a095cfda076980e`.
Rebuilding the current #180 candidate from clean source staging produces exactly
42 modules and passes byte equality in both wheel and clean installation
([identity audit](clean-candidate.json)). Its SHA256 is
`97068955f1f38b2ab0582a3a6806d482f7a7e998b3e9b926f535c99c7b979dd9`.
This is an artifact-identity result; numerical/performance qualification of the
candidate is tracked separately in #180. These wheel binaries are local test
artifacts, not a new published release.

See [package checks](../../docs/development.md#package-checks) for the build and
audit procedure. The existing CI workflow is unchanged.
