"""A benchmark must not hide failed outer/native gates in a successful job."""

from benchmarks.experimental.compare_block import finalize_acceptance


def test_block_benchmark_rejects_incomplete_native_block():
    # The requested k can pass while padding in the native starting block misses
    # its declared stopping bound. The process contract includes the full block.
    report = dict(
        rows=[
            dict(
                quality=dict(accepted=True),
                returned_k_absolute_tolerance_met=True,
                full_block_absolute_tolerance_met=False,
            )
        ]
    )
    assert not finalize_acceptance(report)
    assert report["rows"][0]["physical_and_outer_accepted"]
    assert not report["rows"][0]["native_stopping_accepted"]


def test_block_benchmark_distinguishes_preconditioner_from_inverse():
    report = dict(rows=[dict(quality=dict(accepted=True, preconditioner_cg_stats=[60, 5, 12, 12]))])
    assert finalize_acceptance(report)
    assert report["rows"][0]["accepted"]


def test_block_benchmark_keeps_failed_trials_and_profile():
    report = dict(rows=[dict(quality=dict(accepted=True)), dict(quality=dict(accepted=False))])
    assert not finalize_acceptance(report)
    assert len(report["rows"]) == 2 and report["rows"][0]["accepted"]
    report = dict(
        rows=[dict(quality=dict(accepted=True))], post_profile_quality=dict(accepted=False)
    )
    assert not finalize_acceptance(report)
    assert not finalize_acceptance(dict(rows=[]))


def test_related_starts_rejects_native_or_production_failure():
    from benchmarks.experimental.fem_starts import finalize_starts_acceptance

    report = dict(
        source_quality=dict(accepted=True),
        rows=[
            dict(
                trials=[dict(quality=dict(accepted=True))],
                cupy_trials=[
                    dict(quality=dict(accepted=True), full_block_absolute_tolerance_met=False)
                ],
            )
        ],
    )
    assert not finalize_starts_acceptance(report)
    assert report["rows"][0]["warp_accepted"]
    assert not report["rows"][0]["cupy_accepted"]
    report["rows"][0]["cupy_trials"][0]["full_block_absolute_tolerance_met"] = True
    assert finalize_starts_acceptance(report)
    report["production_baselines"] = [dict(accepted=True, contract_accepted=False)]
    assert not finalize_starts_acceptance(report)
