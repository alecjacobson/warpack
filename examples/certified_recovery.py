"""Bounded, opt-in host orchestration around captured low-mode attempts.

Supply prebuilt graphs and physical output buffers (see docs/completeness.md).
Every solve and verification replay is device-only. This policy driver is NOT
capturable: it explicitly reads results to choose whether to try another graph.
It never assembles/factors matrices or changes a tolerance, gap or mode count.
"""

import time
from dataclasses import dataclass

import warp as wp

from warpack import LowestModeCertificate


@wp.kernel
def _invalid_nullspace(
    residuals: wp.array[wp.float64], tolerance: wp.float64, invalid: wp.array[wp.int32]
):
    i = wp.tid()
    if not wp.isfinite(residuals[i]) or residuals[i] > tolerance:
        wp.atomic_max(invalid, 0, 1)


@dataclass
class Attempt:
    name: str
    graph: object
    values: wp.array
    vectors: wp.array
    status: wp.array
    converged: wp.array
    owner: object = None  # Keep solver/workspaces/backends live through every replay.


def recover(attempts, certificate: LowestModeCertificate):
    """Try a finite list, stopping on certified success or non-retryable failure.

    Keep candidate resources live (e.g. via Attempt.owner).
    K/M, gap and nullspace must stay fixed for the entire call. Output buffers
    in the certificate are distinct from the candidates and retain the last
    attempted result. Inertia setup was already paid by the caller. No new
    factorization/query is hidden in a retry; timings here exclude that setup.

    Only unresolved gap, unconverged residuals or native iteration exhaustion
    allow another attempt. Invalid data/bases/status, inconclusive/unavailable
    inertia, and a count incompatible with output capacity stop immediately.
    A retry may use more starting/guard rows, but must return the same k modes.
    """
    if not isinstance(attempts, (list, tuple)) or not attempts:
        raise ValueError("supply a nonempty finite list/tuple of attempts")
    if wp.get_stream(certificate.device).is_capturing:
        raise RuntimeError("recovery policy reads host status and cannot run during capture")
    for attempt in attempts:
        for source, target in (
            (attempt.values, certificate.values),
            (attempt.vectors, certificate.vectors),
        ):
            if (
                source.shape != target.shape
                or source.dtype != target.dtype
                or source.device != target.device
            ):
                raise ValueError("attempts must provide matching physical eigenpair buffers")
            if source.ptr == target.ptr:
                raise ValueError(
                    "certificate output storage must be distinct from candidate storage"
                )
        for status in (attempt.status, attempt.converged):
            if (
                status.shape != (1,)
                or status.dtype != wp.int32
                or status.device != certificate.device
            ):
                raise ValueError("status and converged must be device int32 scalars")
    certificate.certified.zero_()
    certificate.flags.fill_(certificate.INVALID_DATA)
    report = dict(
        accepted=False, reason="attempts_exhausted", attempts=[], check_capture_seconds=0.0
    )
    if certificate.inertia.status != "available":
        certificate.flags.fill_(
            certificate.COUNT_UNAVAILABLE
            if certificate.inertia.status == "unavailable"
            else certificate.COUNT_INCONCLUSIVE
        )
        report["reason"] = "inertia_" + certificate.inertia.status
        return report
    if certificate.inertia.negative != certificate.capacity:
        certificate.flags.fill_(certificate.COUNT_MISMATCH)
        report["reason"] = "count_requires_new_capacity_or_gap"
        return report
    certificate.values.zero_()
    certificate.vectors.zero_()
    wp.synchronize_device(certificate.device)
    start = time.perf_counter()
    invalid_nullspace = wp.zeros(1, dtype=wp.int32, device=certificate.device)

    def check():
        certificate.run()
        invalid_nullspace.zero_()
        if certificate.q:
            wp.launch(
                _invalid_nullspace,
                certificate.q,
                [certificate.residuals, certificate.residual_tolerance, invalid_nullspace],
                device=certificate.device,
            )

    check()
    with wp.ScopedCapture(device=certificate.device) as cap:
        check()
    wp.synchronize_device(certificate.device)
    report["check_capture_seconds"] = time.perf_counter() - start
    for attempt in attempts:
        certificate.certified.zero_()
        certificate.flags.fill_(certificate.INVALID_DATA)
        start = time.perf_counter()
        wp.capture_launch(attempt.graph)
        wp.copy(certificate.values, attempt.values)
        wp.copy(certificate.vectors, attempt.vectors)
        wp.capture_launch(cap.graph)
        flags = int(certificate.flags.numpy()[0])
        status = int(attempt.status.numpy()[0])
        converged = int(attempt.converged.numpy()[0])
        bad_nullspace = int(invalid_nullspace.numpy()[0])
        row = dict(
            name=attempt.name,
            seconds=time.perf_counter() - start,
            flags=flags,
            status=status,
            converged=converged,
            invalid_nullspace=bad_nullspace,
        )
        report["attempts"].append(row)
        if flags == 0 and status == 0 and converged == certificate.k:
            report.update(accepted=True, reason="certified")
            break
        if status != 0 or not 0 <= converged <= certificate.k:
            report["reason"] = "invalid_solver_status"
            break
        if bad_nullspace:
            report["reason"] = "invalid_nullspace"
            break
        if flags & certificate.COUNT_MISMATCH:
            report["reason"] = "count_requires_new_capacity_or_gap"
            break
        retryable = certificate.GAP_UNRESOLVED | certificate.UNCONVERGED
        if flags & ~retryable:
            report["reason"] = "non_retryable_certificate_failure"
            break
    return report
