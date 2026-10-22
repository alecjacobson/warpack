# Packed-only FSAI promotion attempt

This directory preserves the **unsuccessful first promotion attempt** for
[#139](https://github.com/alecjacobson/warpack/issues/139). It is superseded by the
[scalar-view fix and full Anvil regression](../fsai_setup_regression/README.md).
The older and packed-only environments, rather than the eventual promoted
profile, produced these records.

The candidate is Warp FSAI revision `61077a5c67ea9455cddb614cf68b9bd7e0df9379`,
module SHA256 `aa4f526521c35f87ddfc1610f18bbda395784b636a4a8c844bce8f40ef4ab8b2`.
It includes the newer upstream refit/deferred-factor behavior and the packed
Cholesky workspace, but still reads whole BSR blocks for scalar entry access.

- The complete library suite passed **1,618 tests, five skipped** in 746.76 s
  ([log](full-tests.log)). Four skips concern the older-source-only workspace
  experiment; one is a stock-runtime negative control.
- The built wheel passed the installed-package import and changed-matrix replay
  checks, including the fast path ([report](wheel.json), [log](wheel.log)).
- All six Anvil workers passed numerical checks. Only workers 1 and 2 for each
  profile were uncontended. Their setup + capture cost rose from 2.1879–2.2194 s
  to 3.2592–3.4127 s despite essentially unchanged warm solve time. This triggered
  [#141](https://github.com/alecjacobson/warpack/issues/141) and blocked promotion.
- Workers 3 and `screening/` are retained but **excluded from timing claims**.

Each worker records actual dependency hashes, source hashes, setup components,
accuracy, graph structure and memory scope. `environment.json` describes the
packed-only environment. These historical values must not be relabeled as
measurements of the corrected dependency.
