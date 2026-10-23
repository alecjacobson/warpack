# Shared active-CG packing storage (#165)

The installed device-bucketed CG adapter now shares two maximum-width FP64 packing arrays across mutually exclusive branches. Prefix views retain fixed addresses for replay. Upstream Warp CG states/workspaces remain independent; `for_width` and the public API are unchanged. In-place source preservation, per-application zero padding, branch selection and stopping remain unchanged.

## Complete solve controls

L40, pinned Warp/FSAI, three measured replays after one warmup per fresh worker. Separate/shared pairs have identical source/dependency manifests and exact matrix, reference and initial-vector identities. All 18 measured solves pass the existing native stopping and independent physical/spectrum/subspace gates, and all six external occupancy reports are timing-eligible. No CuPy timings are included in this allocation-only control.

| Problem | Packing bytes, separate → shared | Live Warp pool bytes saved after capture | Setup s, separate → shared | Capture s, separate → shared | Replay median s, separate → shared |
|---|---:|---:|---:|---:|---:|
| [Anvil, 20 modes / width 24](full/anvil-shared-a1.json) | 381,503,808 → 109,001,088 | 272,502,720 | 0.7704 → 0.7825 | 0.8151 → 0.8254 | 1.6578 → 1.6118 |
| [FEM, 100 modes / width 104](full/100-shared-a1.json) | 776,355,840 → 57,507,840 | 718,848,000 | 0.4244 → 0.4168 | 3.8240 → 3.8247 | 2.9833 → 3.0366 |
| [FEM, 200 modes / width 204](full/200-shared-a1.json) | 2,932,899,840 → 112,803,840 | 2,820,096,000 | 0.6268 → 0.6111 | 18.0720 → 18.2738 | 10.4655 → 10.6473 |

This saves approximately **260 MiB on Anvil, 686 MiB for 100 modes, and 2.63 GiB for 200 modes**. The observed live Warp pool decrease exactly matches the requested allocation reduction. Total device free-memory changes also include driver/context/pool behavior; these are recorded separately in every raw report and are not peak-memory measurements. Upstream CG workspaces still dominate much of the remaining wide-solve memory.

Replay changes are small (Anvil −2.8%, 100 modes +1.8%, 200 modes +1.7%) and include variable iteration trajectories from reductions. This change is a memory optimization; these samples do not establish a replay speed improvement. Setup/capture are single measurements per fresh worker, with cached compilation where available, and are not cold-install timings. The README six-backend comparison is not replaced by this narrower control.

## Regression coverage

`tests/test_active_cg_experimental.py` asserts backing-storage sharing and independent upstream CG states, and exercises shrinking/growing/non-leading active sets, zero/scaled RHSs, in-place application, changed values and graph replay for scalar/block matrices and FP32/FP64 working arithmetic. The affected active-CG, release block API, FEM fast replay and geometric preconditioner suite passes **57 tests, 1 skip** ([log](affected-tests.log)). The preceding full-suite result remains separately dated in the main validation documentation.

## Reproduce

Use `python -m benchmarks.experimental.compare_cg_packing --packing-backend separate|shared --public-api ...`; exact remaining arguments, source hashes, runtime versions, input identities and setup scopes are in each JSON file. The private allocation switch exists only for same-source benchmark controls; shared storage is the installed default. All raw JSON, worker logs and external occupancy evidence are retained in `full/`.
