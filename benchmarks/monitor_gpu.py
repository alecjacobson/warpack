"""Run one benchmark with conservative GPU process-occupancy sampling.

Example: python -m benchmarks.monitor_gpu --out build/run.occupancy.json \
    --log build/run.log -- python -m benchmarks.experimental.compare_block ...

Every GPU is monitored. Sampling can miss brief competing jobs; a clean record
is evidence of uncontended execution, not a reservation of the device.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path


def descendants(root):
    """Linux process tree, including subprocess workers used by comparisons."""
    parents = {}
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            # comm can contain spaces and parentheses; split after its final ')'.
            parents[int(stat.parent.name)] = int(stat.read_text().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue  # A process can exit between enumeration and reading.
    own = {root}
    while True:
        expanded = own | {pid for pid, parent in parents.items() if parent in own}
        if expanded == own:
            return own
        own = expanded


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("command", nargs=argparse.REMAINDER)
    args = p.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        p.error("a benchmark command is required")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    samples = []
    with args.log.open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            while True:
                try:
                    own = descendants(process.pid)
                    inventory = subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-compute-apps=pid,process_name",
                            "--format=csv,noheader",
                        ],
                        text=True,
                        timeout=10,
                    ).splitlines()
                    other = [x.strip() for x in inventory if int(x.split(",")[0]) not in own]
                    gpu = subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-gpu=utilization.gpu,memory.used,clocks.sm,temperature.gpu",
                            "--format=csv,noheader",
                        ],
                        text=True,
                        timeout=10,
                    ).strip()
                    error = None
                except (OSError, ValueError, subprocess.SubprocessError) as e:
                    other, gpu, error = [], None, str(e)
                samples.append(dict(time=time.time(), other=other, gpu=gpu, error=error))
                try:
                    code = process.wait(timeout=1)
                    break
                except subprocess.TimeoutExpired:
                    pass
        except BaseException:
            # Only terminate the benchmark this wrapper started.
            process.terminate()
            process.wait()
            raise
    record = dict(
        command=command,
        pid=process.pid,
        exit_code=code,
        timing_eligible=code == 0 and not any(s["other"] or s["error"] for s in samples),
        samples=samples,
        note="Approximately one-second all-GPU process sampling; benchmark descendants excluded. Numerical acceptance must be checked separately.",
    )
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    print(args.out, "exit", code, "timing eligible", record["timing_eligible"], flush=True)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
