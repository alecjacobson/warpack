"""Cost screening of first-metric certification on saved actual search Grams."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.linalg as la
import warp as wp

from benchmarks.anvil_inputs import array_identity
from benchmarks.applications.compare import timed
from benchmarks.audit_profile import graph_summary
from benchmarks.dependency_provenance import warp_dependencies
from benchmarks.experimental.first_metric_certificate import FirstMetricCertificate
from benchmarks.source_provenance import source_manifest
from warpack import SymmetricEigensolver
from warpack.block import Identity, whitening


class Components:
    def __init__(self, n, rank_tol):
        self.source = wp.array(np.eye(n), dtype=wp.float64)
        self.metric = SymmetricEigensolver(Identity(n, self.source.device), n // 2, ncv=n)
        self.metric.which = 1
        self.transform = wp.empty_like(self.source)
        self.rank = wp.zeros(1, dtype=wp.int32)
        self.status = wp.zeros(1, dtype=wp.int32)
        self.active = wp.empty(n, dtype=wp.int32)
        self.cert = FirstMetricCertificate(wp.empty_like(self.source), rank_tol)
        self.rank_tol = rank_tol
        self.graphs, self.audits = {}, {}
        for name, function in [("canonical", self.canonical), ("certificate", self.certificate)]:
            function()
            wp.synchronize()
            with wp.ScopedCapture() as capture:
                function()
            self.graphs[name] = capture.graph
            path = Path("build") / f"first-metric-{n}-{name}.dot"
            wp.capture_debug_dot_print(capture.graph, str(path))
            self.audits[name] = graph_summary(path)
            path.unlink()

    def canonical(self):
        wp.copy(self.metric.g, self.source)
        self.metric.diagonalize()
        self.rank.zero_()
        self.status.zero_()
        wp.launch(
            whitening,
            self.transform.shape,
            [
                self.metric.v,
                self.metric.ritz,
                self.metric.order,
                self.transform,
                self.active,
                self.rank,
                self.status,
                self.rank_tol,
            ],
        )

    def certificate(self):
        wp.copy(self.cert.g, self.source)
        self.cert.apply()

    def check(self, name, original, reference):
        if name == "canonical":
            values, vectors = self.metric.ritz.numpy(), self.metric.v.numpy()
            norm = max(float(la.norm(original)), 1e-300)
            residual = float(la.norm(original @ vectors - vectors * values) / norm)
            orthogonality = float(la.norm(vectors.T @ vectors - np.eye(len(reference))))
            spectrum = float(np.max(abs(np.sort(values) - reference)) / norm)
            rank = int(self.rank.numpy()[0])
            expected = int(np.count_nonzero(reference > self.rank_tol * reference[-1]))
            return dict(
                residual=residual,
                orthogonality=orthogonality,
                spectrum_error=spectrum,
                rank=rank,
                reference_rank=expected,
                accepted=bool(
                    max(residual, orthogonality, spectrum) < 1e-11
                    and self.status.numpy()[0] == 0
                    and rank == expected
                ),
            )
        accepted = bool(self.cert.accepted.numpy()[0])
        delta, lower, upper, product_error = self.cert.diagnostic.numpy().tolist()
        actual_error = None
        valid = True
        if accepted:
            r = self.cert.inverse.numpy().astype(np.longdouble)
            error = r @ original.astype(np.longdouble) @ r.T - np.eye(
                len(reference), dtype=np.longdouble
            )
            actual_error = float(np.max(np.sum(abs(error), axis=1)))
            valid = bool(
                0 < lower <= reference[0]
                and reference[-1] <= upper
                and lower > self.rank_tol * upper
                and actual_error <= delta <= 0.25
            )
        return dict(
            certified_full_rank=accepted,
            delta_upper=delta,
            lambda_lower=lower,
            lambda_upper=upper,
            product_error=product_error,
            extended_precision_congruence_error=actual_error,
            accepted=valid,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--minimum", type=int, default=96)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch", type=int, default=3)
    args = parser.parse_args()
    if args.out.exists() or min(args.repeats, args.batch) < 1:
        parser.error("new destination and positive repeat counts required")
    report = json.loads(args.sample_report.read_text())
    samples = report["first_metric_samples"][0]
    path = Path(samples["snapshot_path"])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == samples["snapshot_sha256"]
    data = np.load(path)
    cache, rows = {}, []
    for sample in samples["rows"]:
        n, index = sample["n"], sample["index"]
        if n < args.minimum:
            continue
        original = data[f"g{index}"]
        reference = la.eigvalsh(original)
        if n not in cache:
            setup, component = timed(lambda: Components(n, 1e-12))
            cache[n] = component
        component = cache[n]
        component.source.assign(original)
        results = {}
        for name in ["canonical", "certificate"]:
            wp.capture_launch(component.graphs[name])
            checks = [component.check(name, original, reference)]
            times = []
            for _ in range(args.repeats):
                seconds, _ = timed(
                    lambda: [wp.capture_launch(component.graphs[name]) for _ in range(args.batch)]
                )
                times.append(seconds / args.batch)
                checks.append(component.check(name, original, reference))
            results[name] = dict(
                samples=times,
                median=float(np.median(times)),
                checks=checks,
                accepted=all(c["accepted"] for c in checks),
            )
        rows.append(
            dict(
                index=index,
                n=n,
                input=array_identity(original),
                zero_diagonal_count=int(np.count_nonzero(np.diag(original) == 0)),
                results=results,
                accepted=all(r["accepted"] for r in results.values()),
            )
        )
        print(
            index,
            n,
            results["canonical"]["median"],
            results["certificate"]["median"],
            results["certificate"]["checks"][0]["certified_full_rank"],
            rows[-1]["accepted"],
            flush=True,
        )
    output = dict(
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        sample_report_sha256=hashlib.sha256(args.sample_report.read_bytes()).hexdigest(),
        snapshot_sha256=samples["snapshot_sha256"],
        rows=rows,
        graph_audits={n: c.audits for n, c in cache.items()},
        accepted=all(r["accepted"] for r in rows),
        source_sha256=source_manifest(__file__),
        dependency=warp_dependencies(),
        protocol="Actual first normalized metric snapshots from the unmodified canonical solve. Each captured component includes restoring its input. Canonical includes rank decision/whitening coefficients; certificate includes factor, inverse, congruence and bounds. Neither includes rotating the physical basis, fallback dispatch or whole-solve trajectory changes. Same-process cached components; three batches of three replays by default. Independent eigen/rank and extended-precision congruence checks after warmup and each batch. Setup/capture excluded; no end-to-end speed claim. External occupancy required.",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    if not output["accepted"]:
        raise SystemExit("Component accuracy failure; evidence retained")


if __name__ == "__main__":
    main()
