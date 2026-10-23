"""Observe first metric certificates without changing canonical solver decisions.

Diagnostic instrumented runs are never performance comparisons. Snapshot storage
and all per-step sampling/certification stay on device; independent host checks
happen only after complete validated solves.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import scipy.linalg as la
import warp as wp

import warpack
from benchmarks.experimental import compare_block
from benchmarks.experimental.first_metric_certificate import FirstMetricCertificate
from benchmarks.source_provenance import source_manifest
from warpack.block import normalize_gram


@wp.kernel(enable_backward=False)
def snapshot(g: wp.array2d[wp.float64], count: wp.array[wp.int32], saved: wp.array3d[wp.float64]):
    i, j = wp.tid()
    if count[0] < saved.shape[0]:
        saved[count[0], i, j] = g[i, j]


@wp.kernel(enable_backward=False)
def record(
    count: wp.array[wp.int32],
    n: int,
    accepted: wp.array[wp.int32],
    diagnostic: wp.array[wp.float64],
    rows: wp.array2d[wp.float64],
):
    index = count[0]
    if index < rows.shape[0]:
        rows[index, 0] = wp.float64(n)
        rows[index, 1] = wp.float64(accepted[0])
        for j in range(4):
            rows[index, j + 2] = diagnostic[j]
    count[0] = index + 1


class SampleFirstMetrics(warpack.BlockLOBPCG):
    sample_capacity = 96

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        branches = [self] + ([] if self._search is None else self._search.buckets)
        size = max(b.metric.g.shape[0] for b in branches)
        self._sample_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._sample_grams = wp.empty(
            (self.sample_capacity, size, size), dtype=wp.float64, device=self.device
        )
        self._sample_rows = wp.empty(
            (self.sample_capacity, 6), dtype=wp.float64, device=self.device
        )
        for branch in branches:
            branch._first_certificate = None
            if 2 <= branch.metric.g.shape[0] <= 512:
                branch._first_certificate = FirstMetricCertificate(branch.metric.g, branch.rank_tol)
            branch._sample_count = self._sample_count
            branch._sample_grams = self._sample_grams
            branch._sample_rows = self._sample_rows

    def initialize(self, seed=42):
        self._sample_count.zero_()
        super().initialize(seed)

    def launch(self, kernel, shape, args):
        super().launch(kernel, shape, args)
        if (
            kernel is normalize_gram
            and args[0] is self.metric.g
            and wp.get_stream(self.device).is_capturing
            and self._first_certificate is not None
        ):
            cert = self._first_certificate
            cert.apply()
            wp.launch(
                snapshot,
                self.metric.g.shape,
                [self.metric.g, self._sample_count, self._sample_grams],
                device=self.device,
            )
            wp.launch(
                record,
                1,
                [self._sample_count, cert.n, cert.accepted, cert.diagnostic, self._sample_rows],
                device=self.device,
            )

    def samples(self, path):
        count = int(self._sample_count.numpy()[0])
        rows = self._sample_rows.numpy()[: min(count, self.sample_capacity)]
        grams = self._sample_grams.numpy()
        saved, result = {}, []
        for i, row in enumerate(rows):
            n, accepted = int(row[0]), bool(row[1])
            g = grams[i, :n, :n].copy()
            saved[f"g{i}"] = g
            values = la.eigvalsh(g)
            low, high = float(values[0]), float(values[-1])
            rank = int(np.count_nonzero(values > self.rank_tol * high))
            delta, lower, upper, product_error = row[2:].tolist()
            qualified = not accepted or (
                rank == n and 0 < lower <= low and high <= upper and delta <= 0.25
            )
            result.append(
                dict(
                    index=i,
                    n=n,
                    accepted=accepted,
                    reference_rank=rank,
                    minimum_eigenvalue=low,
                    maximum_eigenvalue=high,
                    delta_upper=delta,
                    lambda_lower=lower,
                    lambda_upper=upper,
                    product_error=product_error,
                    independent_certificate_check=qualified,
                )
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **saved)
        return dict(
            observed_count=count,
            stored_count=len(result),
            rows=result,
            accepted_count=sum(r["accepted"] for r in result),
            all_certificates_verified=all(r["independent_certificate_check"] for r in result),
            snapshot_path=str(path),
            snapshot_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--snapshots", type=Path, required=True)
    choice, remainder = parser.parse_known_args()
    destination_parser = argparse.ArgumentParser(add_help=False)
    destination_parser.add_argument("--out", type=Path, required=True)
    destination, _ = destination_parser.parse_known_args(remainder)
    if "--public-api" not in remainder:
        parser.error("the public block composition is required")
    original, manifest, argv = warpack.BlockLOBPCG, compare_block.source_manifest, sys.argv
    instances, error = [], None

    class Tracking(SampleFirstMetrics):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            instances.append(self)

    try:
        warpack.BlockLOBPCG = Tracking
        compare_block.source_manifest = lambda *_: source_manifest(__file__)
        sys.argv = [sys.argv[0]] + remainder
        compare_block.main()
    except SystemExit as caught:
        error = caught
    finally:
        warpack.BlockLOBPCG, compare_block.source_manifest, sys.argv = original, manifest, argv
    if destination.out.exists():
        report = json.loads(destination.out.read_text())
        report["first_metric_samples"] = [
            instance.samples(choice.snapshots.with_name(f"{choice.snapshots.stem}-{i}.npz"))
            for i, instance in enumerate(instances)
        ]
        report["timing_eligible"] = False
        report["timing_note"] = (
            "Instrumented canonical solve; certificate/snapshot overhead included. Do not use as a solver timing."
        )
        report["first_metric_scope"] = (
            "Last replay only. Observe the original normalized first Gram and candidate certificate; canonical rank/whitening decisions are unchanged. Independent SciPy spectral checks after complete solve."
        )
        report["accepted"] &= all(
            s["all_certificates_verified"] for s in report["first_metric_samples"]
        )
        destination.out.write_text(json.dumps(report, indent=2) + "\n")
        if not report["accepted"]:
            raise SystemExit(1)
    if error is not None:
        raise SystemExit(error.code)


if __name__ == "__main__":
    main()
