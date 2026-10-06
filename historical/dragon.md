# Dragon showcases, mesh validation, and elasticity audit

Archived measurements and experiment notes from the README before its documentation cleanup. Results, test counts, and environment descriptions refer to the runs recorded here. Run commands from the repository root.

[Current README](../README.md) · [Historical index](README.md)

The [v0.1.2 GIF](https://github.com/alecjacobson/warpack/releases/tag/v0.1.2) uses the supplied [`xyzrgb_dragon-720K-ftetwild.mesh`](../xyzrgb_dragon-720K-ftetwild.mesh): **24,776 vertices, 93,354 tetrahedra, and 74,328 degrees of freedom**. The tetrahedra are read directly from that file; this project does not invoke a mesher. Despite its filename, this volume mesh has 93,354 tets. No mesh decimation or modal reduction is used in the solve.

The original `dragon-H/dragon.mesh` has **330,206 vertices, 1,187,670 tetrahedra, and 990,618 degrees of freedom**, with **3,881,370 stiffness blocks**. The [historical full-dragon performance measurements](v0.1.1-optimization.md) refer to that original, larger mesh. Its animation remains in the [v0.1.1 release](https://github.com/alecjacobson/warpack/releases/tag/v0.1.1).

The reference geometry is normalized to a longest bounding-box extent of one metre. The material is `E = 100,000 Pa`, `ν = 0.3`, and `ρ = 1,000 kg/m³`. The polynomial stable Neo-Hookean energy is

```text
ψ(F) = μ/2 (tr(FᵀF) - 3) + (λ+μ)/2 (det(F) - α)²
α = 1 + μ/(λ+μ)
```

Here λ and μ are the physical Lamé constants. The assembled Hessian is the stress-free tangent at `F=I`, checked independently by finite differences of this energy. Mass is **lumped**, not consistent. The body is free: six rigid modes are checked and omitted from the film. We solve `M⁻¹/² H M⁻¹/²`, using a positive unit shift for the inverse, then recover mass-orthonormal physical displacements. The first elastic frequency is **0.90917 Hz** for fTetWild, versus **0.89756 Hz** for the original mesh.

The v0.1.2 dragon GIF exercises all 20 elastic modes in order, from rest to peak amplitude and back using gptoolbox's `squease` function. Peak poses are scaled independently to fit within a centered box twice the original dimensions, with 5% margin; the entire animation occupies at most **1.95×** the original box dimensions. The camera stays fixed. The exact gptoolbox jet-range palette is `okloop(256,-4*pi/3,-pi/2)`, with Polyscope-style alternating scalar stripes. Color represents **instantaneous displacement norm**, divided by the largest displayed displacement anywhere in the whole sequence; the color scale stays fixed as each mode grows and returns to rest. See the [style/scale metadata](../results/dragon_ftetwild_teaser.json) and [evaluated scene validation](../results/dragon_ftetwild_teaser_validation.json).

These are exaggerated linear mode shapes, not nonlinear deformation trajectories. At the requested large display amplitudes, some tetrahedra invert on both meshes; this does not indicate an invalid rest mesh or an eigensolver error.

The original v0.1.0 film has a white studio background and displacement-magnitude pseudocolor, normalized separately per mode. Each mode plays one cycle in two seconds, with its physical frequency labelled. The maximum displayed surface displacement is 2.5% of body length. An analytic cubic-volume check over the **entire sinusoidal cycle**, on every tetrahedron in every mode, found no inversions. Display amplitude and playback speed are illustration choices, not a physical forcing amplitude or real-time playback.

```bash
# The original workspace keeps this dataset in a sibling directory.
python examples/dragon_modes.py \
  --mesh ../bbw-comparison/dragon-H/dragon.mesh \
  --method krylov --width 64 --iterations 1

# Blender 4.5 is used for the published scene.
blender -b --python examples/render_modes.py -- --preview --render
python examples/validate_animation.py
```

To reproduce the historical v0.1.2 dragon teaser with Blender 4.5 and [gifski](https://gif.ski/):

```bash
python -m examples.dragon_modes --mesh xyzrgb_dragon-720K-ftetwild.mesh \
  --out results/dragon_ftetwild_modes.npz
blender -b --python examples/render_modes.py -- \
  --input results/dragon_ftetwild_modes.npz --up-axis Z \
  --output results/dragon_ftetwild_base.blend
blender -b --python examples/render_teaser.py -- \
  --base results/dragon_ftetwild_base.blend --input results/dragon_ftetwild_modes.npz \
  --output results/dragon_ftetwild_teaser.blend --up-axis Z --stripes \
  --frames build/ftetwild_frames --render
blender -b results/dragon_ftetwild_teaser.blend --python examples/validate_teaser.py
gifski --fps 20 --quality 85 --width 720 --repeat 0 \
  --output results/dragon_ftetwild_teaser.gif build/ftetwild_frames/frame_*.png
```

The GIF contains 33 frames per mode at 20 fps (33 seconds total), on a white studio background. The editable teaser scene has baked geometry and color-amplitude curves and a packed colormap, so it does not depend on custom Python drivers.

For the original benchmark mesh, download `dragon.mesh.gz` from the [v0.1.0 release](https://github.com/alecjacobson/warpack/releases/tag/v0.1.0) and decompress it to `../bbw-comparison/dragon-H/dragon.mesh`. That release includes the exact input mesh, the mass-normalized numerical results and physical modes, and the editable `.blend` scene. Dataset provenance and licensing are in [THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES).

## Dragon mesh quality and independent mode verification

Both inputs have one connected component, no orphan vertices, and no zero-volume elements. The new mesh has substantially better element shapes:

| Metric | Original dragon-H | Supplied fTetWild |
|---|---:|---:|
| Minimum dihedral angle | 0.492° | 10.471° |
| Minimum mean-ratio quality (regular tet = 1) | 0.00405 | 0.37176 |
| Median mean-ratio quality | 0.6533 | 0.8277 |
| Worst regular-reference Frobenius condition (regular tet = 1) | 132.19 | 3.68 |
| Tets with mean ratio below 0.1 | 58 | 0 |

The original mesh's 58 lowest-quality tets contribute at most **0.057%** of any of the first 20 elastic modes' strain energy. This argues against those elements causing grossly incorrect low modes. See the [element-quality report](../results/mesh_quality.json) and [element-energy and display-Jacobian checks](../results/dragon_mesh_physics.json).

For **each mesh**, Warp was compared with CuPy `eigsh`, SciPy ARPACK, and C++ Spectra on the identical mass-normalized matrix, initial vector, 26 requested eigenpairs, and 64-vector subspace. Warp and CuPy share cuDSS factors; ARPACK and Spectra share a separate CPU SuperLU factorization. Spectra uses a reference-only callback to that factorization. This is accuracy validation, not a timing benchmark.

Across both meshes and all three references, the maximum relative elastic eigenvalue difference is **5.88e-12**. All 20 elastic mode shapes have mass-weighted, sign-invariant modal assurance equal to **1 to floating-point roundoff**; the six-dimensional rigid subspaces also coincide to roundoff. The largest original-problem normalized residual is **4.48e-8**, including rigid modes. Raw checks: [original mesh](../results/dragon_original_solver_validation.json), [fTetWild](../results/dragon_ftetwild_solver_validation.json).

The first 20 frequencies change by **−0.43% to +4.16%** between meshes. The new mesh is much coarser and has slightly different geometry, so this comparison does **not** establish continuum convergence or attribute the frequency changes solely to element quality. We use fTetWild for its better element shapes; the original modes also solve their discrete eigenproblem correctly.

To reproduce, install the reference dependencies and build against the pinned Spectra checkout in [Validation and benchmarking](../README.md#validation-and-benchmarking):

```bash
g++ -O3 -DNDEBUG -shared -fPIC -I/usr/include/eigen3 -Ibuild/spectra/include \
  benchmarks/spectra_callback.cpp -o build/spectra_callback.so
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python -m benchmarks.validate_dragon_meshes \
  xyzrgb_dragon-720K-ftetwild.mesh --out results/dragon_ftetwild_solver_validation.json
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python -m benchmarks.validate_dragon_meshes \
  ../bbw-comparison/dragon-H/dragon.mesh --out results/dragon_original_solver_validation.json
OPENBLAS_NUM_THREADS=4 python -m benchmarks.mesh_quality \
  ../bbw-comparison/dragon-H/dragon.mesh xyzrgb_dragon-720K-ftetwild.mesh
# Requires both numerical NPZs and the original/current teaser metadata:
OPENBLAS_NUM_THREADS=4 python -m benchmarks.dragon_mesh_physics
```

### Mode 09: constitutive and mass audit

Mode 09's large whisker motion survives an independent **StVK** assembly and a **consistent-mass** solve. We found no rest-Hessian error. The audit uses face-cross-product shape gradients and Green-strain directional derivatives, independently of the production inverse-reference-matrix/block formula. Both full meshes, including graph-reassembled production stiffness, agree with this reference to **4.0e-16 relative Frobenius error**. Recomputed frequencies differ by at most **7.2e-13 relative**, and all 20 elastic mode shapes agree to roundoff.

At the stress-free rest state, StVK, ordinary log Neo-Hookean, and the implemented polynomial stable Neo-Hookean energy have the same tangent for matching physical Lamé constants:

```text
P(I) = 0
DP(I)[D] = μ (D + Dᵀ) + λ tr(D) I
H_ij = V [ μ (g_i · g_j) I + λ g_i g_jᵀ + μ g_j g_iᵀ ]
```

Here `g_i` is the reference shape-function gradient. The polynomial energy's volumetric coefficient is **λ+μ**, not the physical λ alone. This is the parameter conversion in §3.4 of [Smith, de Goes, and Kim (2018)](https://www.tkim.graphics/NEO/StableNeoHookean2018.pdf). Our formula is the polynomial variant, without the paper's optional regularized origin-barrier term; correctly calibrating that variant also gives the same rest tangent. Changing the energy family therefore does not change these infinitesimal rest modes. Finite-amplitude nonlinear trajectories or modes about a predeformed equilibrium are different problems.

Complex-step checks independently differentiate the energy to obtain stress, then differentiate nodal energy gradients to check every element-Hessian entry. They cover all four energy variants, skewed tets, reversed vertex ordering, and Poisson ratios −0.2, 0.3, and 0.49. A separate exact tetrahedral quadrature check verifies consistent mass and its lumped row sums. **102 tests pass**; see [audit validation](../results/elasticity_audit_validation.json), [test source](../tests/test_elasticity_reference.py), and [raw full-mesh results](../results/elasticity_audit.json).

On fTetWild, switching from lumped to consistent mass changes Mode 09 from **1.42311 to 1.42603 Hz (+0.205%)**, with mode-shape MAC **0.9999946**. Its peak/RMS displacement ratio remains **52.6**. In an explicitly defined region within 0.08 body lengths along mesh edges from the tip, **0.0572% of total mass carries 73.8% of this mode's kinetic energy**. The original mesh shows the same localization (0.0528% of mass, 60.9% of modal kinetic energy). Both meshes have one face-connected tetrahedral component and no faces shared by more than two tets. These findings support a localized vibration of a slender appendage in the discrete elastic model, rather than a constitutive, lumping, or disconnected-element artifact. They do not establish continuum convergence at the appendage.

The v0.1.2 dragon teaser's bounding-box-based display scaling moves this tip by **76.8% of the entire body's length**. A twice-size bounding box is a weak amplitude limit for a thin feature: this is **30.7×** a 2.5%-of-body-length display cap. The figure below changes only the display amplitude of the same verified eigenvector; the large pose should not be interpreted as a nonlinear deformation prediction.

![Mode 09 at rest, with a 2.5% displacement cap, and with the historical dragon README amplitude](../results/mode09_amplitude_comparison.png)

Reproduce after generating both mode archives:

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python -m benchmarks.audit_elasticity
OPENBLAS_NUM_THREADS=4 python -m pytest -q tests/test_elasticity_reference.py
python -m examples.plot_mode09_audit
```

The alternate StVK and consistent-mass assembly and solves use Warp kernels and cuDSS. Host NumPy/SciPy are used only for explicit audit diagnostics and reference differentiation, not production eigensolver computation.
