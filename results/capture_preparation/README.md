# Graph-preparation attribution (#168)

On the 200-mode/width 204 FEM fixture with compact projected graphs and shared
adapter storage, the diagnostic attributes the 12.6497-second capture phase as:

| Phase | Seconds |
|---|---:|
| Initial eager solve preparation | 1.1198 |
| Active-search warmup, all 51 buckets | 11.0315 |
| Remaining outer capture/setup | 0.4985 |

[Raw profile](200-compact-a1.json) records each bucket separately, growing from
about 0.18 to 0.28 seconds as projected dimensions grow. Bucket times are nested
inside the 11.0315-second warmup and must not be added to it again. Module loading
and lazy child-graph preparation remain charged to the phase where they occur.
This adds synchronization around eager phases for attribution; it is not a
CUDA API trace or an uninstrumented performance baseline.

The first output and all three subsequent complete replays pass native stopping
and independent physical/spectrum/subspace checks. External GPU occupancy is
clean. The captured graph retains device-only nodes/transfers.

`ActiveSearch.warm()` already sets the active residual count to zero but retains
the previous search-history vectors. Every branch still performs numerical
Rayleigh–Ritz preparation. Clearing that setup-only history before branch warmup
may reduce dense numerical work without skipping any branch or kernel preparation.
The captured solve initializes its own search space afterward. This proposed
change requires full replay, callback, spectrum and timing regression controls;
it is not yet installed.
