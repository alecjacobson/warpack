"""Run the candidate revision's unchanged upstream FSAI suite in an isolated worker."""

import argparse
import hashlib
import importlib.util
import inspect
import sys
import unittest
from pathlib import Path
from urllib.request import urlopen

import warp as wp
import warp._src.optim as optim
from warp.optim import linear

from benchmarks.experimental.fsai_current import REVISION, load

TEST_SHA256 = "f2082cfb751d4cb672894baab1f3575f02f47e8f9f940b5424990fbb827d6478"
PROMOTED_TEST_SHA256 = "8c8e5156c210312111e10373efd88128c8b3e5d7a7d36d5b8f49083ad94b4343"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packed", action="store_true")
    parser.add_argument(
        "--installed", action="store_true", help="Verify and test the current installer profile"
    )
    parser.add_argument("--block-access", action="store_true")
    parser.add_argument("--block-access-layout", choices=("flat", "view"), default="flat")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    if args.installed and (args.packed or args.block_access):
        parser.error("--installed cannot be combined with experimental source patches")
    revision, test_digest = REVISION, TEST_SHA256
    path = Path("build/current_test_fsai.py")
    if args.installed:
        from scripts.install_fsai_overlay import PROFILES

        revision, module_digest = PROFILES["current"]
        module_path = Path(inspect.getfile(linear.FSAI))
        if hashlib.sha256(module_path.read_bytes()).hexdigest() != module_digest:
            raise ValueError("installed FSAI differs from the current installer profile")
        test_digest = PROMOTED_TEST_SHA256
        path = Path("build/promoted_test_fsai.py")
    if not path.exists() and args.download:
        url = f"https://raw.githubusercontent.com/alecjacobson/warp/{revision}/warp/tests/test_fsai.py"
        with urlopen(url, timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != test_digest:
            raise ValueError("upstream test checksum mismatch")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    if hashlib.sha256(path.read_bytes()).hexdigest() != test_digest:
        raise ValueError("upstream test checksum mismatch")
    if args.cache:
        wp.config.kernel_cache_dir = str(args.cache)
    module = (
        sys.modules[linear.FSAI.__module__]
        if args.installed
        else load(download=args.download, packed=args.packed)
    )
    if args.block_access:
        from benchmarks.experimental.fsai_block_access import load as load_block_access

        module = load_block_access(args.block_access_layout)
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
