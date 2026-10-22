"""Run the candidate revision's unchanged upstream FSAI suite in an isolated worker."""

import argparse
import hashlib
import importlib.util
import sys
import unittest
from pathlib import Path
from urllib.request import urlopen

import warp as wp
import warp._src.optim as optim
from warp.optim import linear

from benchmarks.experimental.fsai_current import REVISION, load

TEST_SHA256 = "f2082cfb751d4cb672894baab1f3575f02f47e8f9f940b5424990fbb827d6478"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packed", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    path = Path("build/current_test_fsai.py")
    if not path.exists() and args.download:
        url = f"https://raw.githubusercontent.com/alecjacobson/warp/{REVISION}/warp/tests/test_fsai.py"
        with urlopen(url, timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != TEST_SHA256:
            raise ValueError("upstream test checksum mismatch")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    if hashlib.sha256(path.read_bytes()).hexdigest() != TEST_SHA256:
        raise ValueError("upstream test checksum mismatch")
    if args.cache:
        wp.config.kernel_cache_dir = str(args.cache)
    module = load(download=args.download, packed=args.packed)
    # Process-local import bindings only; never write installed dependencies.
    optim.fsai = module
    linear.FSAI = module.FSAI
    sys.modules["warp._src.optim.fsai"] = module
    spec = importlib.util.spec_from_file_location("warpack_candidate_fsai_tests", path)
    tests = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = tests
    spec.loader.exec_module(tests)
    suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
