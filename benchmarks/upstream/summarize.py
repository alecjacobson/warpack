"""Summarize recorded upstream runs without executing any numerical work."""

import ast
import collections
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    return json.loads((ROOT / "results" / name).read_text())


def state(t):
    if t.get("expected_failure"):
        return "xpassed" if t["outcome"] == "passed" else "xfailed"
    return t["outcome"]


def main():
    result = {}
    warp = load("upstream_cupy.json")["tests"]
    native = load("upstream_cupy_native.json")["tests"]
    assert len(warp) == len(native) == 3984, "CuPy inventory is incomplete"
    assert {t["nodeid"] for t in warp} == {t["nodeid"] for t in native}
    for name, rows in [("warp", warp), ("native_overlay", native)]:
        groups = collections.defaultdict(collections.Counter)
        for t in rows:
            groups[t["nodeid"].split("::")[1]][state(t)] += 1
        result["cupy_" + name] = dict(
            counts=dict(collections.Counter(state(t) for t in rows)), by_class=dict(groups)
        )
        skips = collections.Counter()
        for t in rows:
            if state(t) == "skipped":
                try:
                    reason = ast.literal_eval(t["detail"])[-1]
                except (SyntaxError, ValueError, TypeError):
                    reason = t["detail"]
                skips[reason] += 1
        result["cupy_" + name]["skip_reasons"] = dict(skips)
    reference = {t["nodeid"]: state(t) for t in native}
    result["cupy_warp_failures"] = [
        dict(nodeid=t["nodeid"], native=reference.get(t["nodeid"]), detail=t["detail"])
        for t in warp
        if state(t) == "failed"
    ]
    spectra = load("upstream_spectra_warp.json")["cases"]
    assert len(spectra) == len({r["id"] for r in spectra}) == 360
    groups = collections.defaultdict(collections.Counter)
    for row in spectra:
        groups[row["family"]][row["outcome"]] += 1
    result["spectra_warp"] = dict(
        counts=dict(collections.Counter(r["outcome"] for r in spectra)),
        by_family=dict(groups),
        failed_upstream_allows_nonconvergence=sum(
            r["outcome"] == "failed" and r["upstream_allows_nonconvergence"] for r in spectra
        ),
        failed_original_residual_passes=sum(
            r["outcome"] == "failed"
            and r.get("max_absolute_residual", float("inf"))
            <= r.get("upstream_absolute_tolerance", 0)
            for r in spectra
        ),
        failed_only_internal_convergence=sum(
            r["outcome"] == "failed"
            and r.get("max_absolute_residual", float("inf"))
            <= r.get("upstream_absolute_tolerance", 0)
            and r.get("spectrum_error", 0) <= r.get("upstream_absolute_tolerance", 0)
            and r.get("reported_converged", r["k"]) < r["k"]
            for r in spectra
        ),
        failure_ids=[r["id"] for r in spectra if r["outcome"] == "failed"],
        timeout_ids=[r["id"] for r in spectra if r["outcome"] == "timeout"],
    )
    benchmarks = load("upstream_benchmarks.json")["cases"]
    assert len(benchmarks) == len({r["id"] for r in benchmarks}) == 40
    groups = collections.defaultdict(collections.Counter)
    medians = collections.defaultdict(list)
    residuals = collections.defaultdict(list)
    for row in benchmarks:
        key = row["family"] + "_" + str(row["n"])
        if row.get("outcome"):
            groups[key]["worker_" + row["outcome"]] += 1
        else:
            for backend in row["results"]:
                groups[key][backend["backend"] + "_" + backend["outcome"]] += 1
                if backend["outcome"] == "passed":
                    medians[key + "_" + backend["backend"]].append(backend["solve_seconds"] * 1000)
                    residuals[key + "_" + backend["backend"]].append(
                        backend["max_absolute_residual"]
                    )
    result["benchmarks"] = dict(groups)
    result["gpu_benchmark_medians"] = {
        key: dict(
            accepted_datasets=len(values),
            median_ms=statistics.median(values),
            max_absolute_residual=max(residuals[key]),
        )
        for key, values in medians.items()
    }
    native_groups = collections.defaultdict(list)
    for row in load("upstream_provenance.json")["native_spectra_benchmark"]["rows"]:
        for backend in ("arpack", "spectra"):
            native_groups[(row["family"], row["n"], row["dataset"], backend)].append(
                row[backend + "_ms"]
            )
    medians = collections.defaultdict(list)
    for (family, n, _, backend), samples in native_groups.items():
        assert len(samples) == 10
        medians[f"{family}_{n}_{backend}"].append(statistics.median(samples))
    result["native_benchmark_medians"] = {
        key: dict(datasets=len(values), median_ms=statistics.median(values))
        for key, values in medians.items()
    }
    (ROOT / "results/upstream_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    for name, info in result.items():
        if name != "cupy_warp_failures":
            print(name, json.dumps(info))


if __name__ == "__main__":
    main()
