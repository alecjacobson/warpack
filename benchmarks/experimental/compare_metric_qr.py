"""Complete FEM controls for explicitly guarded Ritz-basis reuse (#183)."""

import argparse
import json
import sys
from pathlib import Path

import warpack
from benchmarks.cached_fem_reference import FEMReferenceCache
from benchmarks.experimental import compare_block
from benchmarks.experimental.metric_qr_reuse import MetricQRReuseBlock
from benchmarks.source_provenance import source_manifest


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--metric-defect", type=float, choices=[0, 1e-14, 1e-13, 1e-12], required=True
    )
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--compact-graphs", action="store_true")
    parser.add_argument("--wide-only", action="store_true")
    parser.add_argument("--installed-default", action="store_true")
    choice, remainder = parser.parse_known_args()
    destination_parser = argparse.ArgumentParser(add_help=False)
    destination_parser.add_argument("--out", type=Path, required=True)
    destination, _ = destination_parser.parse_known_args(remainder)
    if "--public-api" not in remainder:
        parser.error("this control requires the public Warp composition")
    if "--cupy" in remainder and not choice.installed_default:
        parser.error("CuPy reference requires --installed-default")
    if choice.installed_default and choice.metric_defect:
        parser.error("--installed-default requires --metric-defect 0")
    original, manifest, prepare, reference, argv = (
        warpack.BlockLOBPCG,
        compare_block.source_manifest,
        compare_block.prepare,
        compare_block.cupy_trials,
        sys.argv,
    )
    base = MetricQRReuseBlock if choice.metric_defect else original
    instances = []
    error = None
    reference_active = False

    class Tracking(base):
        _ritz_qr_enabled = choice.installed_default
        _metric_reuse_tolerance = choice.metric_defect
        _metric_reuse_minimum = 96 if choice.wide_only else 1
        _metric_reuse_maximum = 256 if choice.wide_only else 2147483647

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs, compact_projected_graphs=choice.compact_graphs)
            instances.append(self)

    def tracked_prepare(*args, **kwargs):
        result = prepare(*args, **kwargs)
        check = result[-1]

        def tracked_check(*values, **options):
            quality = check(*values, **options)
            if (
                not reference_active
                and instances
                and (choice.metric_defect or choice.installed_default)
            ):
                usage = (
                    instances[-1]._ritz_qr_usage
                    if choice.installed_default
                    else instances[-1].metric_reuse_usage
                )
                quality["metric_qr_reuse_and_fallback"] = (
                    None if usage is None else usage.numpy().tolist()
                )
            return quality

        return (*result[:-1], tracked_check)

    def tracked_reference(*args, **kwargs):
        nonlocal reference_active
        previous = reference_active
        reference_active = True
        try:
            return reference(*args, **kwargs)
        finally:
            reference_active = previous

    with FEMReferenceCache(choice.reference_cache) as cache:
        try:
            warpack.BlockLOBPCG = Tracking
            compare_block.source_manifest = lambda *_: source_manifest(__file__)
            compare_block.prepare = tracked_prepare
            compare_block.cupy_trials = tracked_reference
            sys.argv = [sys.argv[0]] + remainder
            compare_block.main()
        except SystemExit as caught:
            error = caught
        finally:
            (
                warpack.BlockLOBPCG,
                compare_block.source_manifest,
                compare_block.prepare,
                compare_block.cupy_trials,
                sys.argv,
            ) = (
                original,
                manifest,
                prepare,
                reference,
                argv,
            )
    if destination.out.exists():
        report = json.loads(destination.out.read_text())
        report["args"].update(
            metric_defect=choice.metric_defect,
            compact_graphs=choice.compact_graphs,
            reference_cache=str(choice.reference_cache),
            wide_only=choice.wide_only,
            installed_default=choice.installed_default,
        )
        report["independent_reference_cache"] = cache.records
        report["metric_qr_scope"] = (
            "Explicit complete two-pass QR control (metric-defect 0), installed default "
            "(--installed-default), or benchmark-only "
            "fresh computed-metric defect check. The latter reuses X and cached AX "
            "only after a finite device check, retaining fresh MX; other cases run "
            "the complete original QR. Computed-metric bounds do not certify "
            "exact-arithmetic physical products. All original physical/native/rank "
            "gates remain. Conditional fallback is restricted to owned Warp mass "
            "operators. Each worker owns its preconditioner/solver; only independently "
            "verified CPU references are reused outside timing. Counters are read "
            "during post-solve validation, outside compute timing."
            " With --wide-only, widths outside96..256 retain original QR with "
            "no added control arrays or graph nodes. Installed default additionally guards "
            "Ada SM89, chunk4096, tol>=1e-11, image reuse and symmetric products. "
            "With --cupy, only the installed-default Warp policy is compared. "
            "CuPy uses the supplied Warp preconditioner/base CG, so its adapter setup "
            "excludes those shared factors and is not an independent total setup cost. "
            "QR usage counters are attached only to Warp validation, never CuPy rows."
        )
        destination.out.write_text(json.dumps(report, indent=2) + "\n")
    if error is not None:
        raise SystemExit(error.code)


if __name__ == "__main__":
    main()
