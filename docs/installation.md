# Installation

[Back to README](../README.md)

warpack requires Python >=3.10, Warp >=1.15 and an NVIDIA CUDA GPU. Conditional
graphs need a driver supporting CUDA 12.4 or later. Choose a Warp native build
compatible with the installed driver. The release was tested on Linux x86-64,
NVIDIA L40, driver 570.158.01, with the CUDA 12 build of Warp described below.
Stock Warp 1.15 has separate compatibility coverage; this does not certify every
later Warp/runtime combination.

## Released library

```bash
python -m pip install 'https://github.com/alecjacobson/warpack/releases/download/v0.2.0/warpack-0.2.0-py3-none-any.whl'
```

[Release assets](https://github.com/alecjacobson/warpack/releases/tag/v0.2.0)
include the wheel, source distribution and SHA256 checksums. The source
distribution builds the library. Use a repository checkout for the full
examples, benchmark harnesses, meshes and reference fixtures.

## Source checkout and optional dependencies

```bash
git clone https://github.com/alecjacobson/warpack.git
cd warpack
python -m pip install -e '.[test]'
```

| Extra | Purpose |
|---|---|
| `test` | pytest and SciPy for tests and independent references |
| `cudss` | Pinned cuDSS 0.8.0.10 and Warp binding for sparse direct inverses |
| `benchmark` | SciPy, CuPy and matplotlib for reference comparisons |

Install extras together with `python -m pip install -e '.[test,cudss,benchmark]'`.
The cuDSS extra uses the CUDA 12 package and a pinned Git revision of
[warp-cuDSS](https://github.com/alecjacobson/warp-cuDSS).
It does not install or change a GPU driver. NumPy is a package dependency used
by Warp's array interface; NumPy/SciPy/CuPy do not compute library eigenpairs.

## Fast block solves

`BatchedCGPreconditioner` and the tuned Anvil example require the upstream
batched-CG/FSAI implementation pinned for v0.2.0. In particular, stock Warp 1.15
has `batch_offsets` but lacks the required `max_batch_length`; the adapter
rejects that configuration during setup. Installing a warpack extra does not
install the overlay.

From the repository root:

```bash
python -m pip install --no-deps --target build/warp-fsai \
  'https://github.com/NVIDIA/warp/releases/download/v1.18.0/warp_lang-1.18.0%2Bcu12-py3-none-manylinux_2_28_x86_64.whl'
python scripts/install_fsai_overlay.py build/warp-fsai
PYTHONPATH="$PWD/build/warp-fsai:$PWD" OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python -m examples.anvil_block --out build/anvil-block.json
```

The installer verifies the target version and all three file checksums before
writing unmodified upstream modules from revision
`04427b997fe7ff07bc6ed1dde5ca9813496c6b55`. The separate target and per-command
`PYTHONPATH` leave the default Warp installation unchanged. Reuse that
`PYTHONPATH` for fast-path scripts and the full pinned test suite.

See the [release guide](releases/v0.2.0.md) for the exact public configuration,
resource ownership and recorded memory costs. FSAI construction is setup work;
its application and the eigensolver replay run on device.

## Check the environment

```bash
python -c 'import warp; print(warp.__version__); print(warp.__file__)'
python -m examples.block_reuse
```

The bar example uses stock-compatible diagonal preconditioning and SciPy only
for independent validation after solving. For the fast environment, prefix
commands with the overlay's `PYTHONPATH`. Missing upstream fast-path features
should produce a setup error; do not interpret them as eigensolver nonconvergence.
