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

The [history-clearing control](zero-history/README.md) saved little: most of
this cost remained. The [direct-recording prototype](capture-only/README.md)
removed redundant numerical branch solves, including cold-cache qualification
on both supported Warp environments. See the [production controls](production/README.md)
for the guarded implementation and current setup/capture/replay measurements.
