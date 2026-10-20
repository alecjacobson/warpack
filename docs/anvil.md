# Anvil visualization

[`examples/problems/elasticity.py`](../examples/problems/elasticity.py) assembles the stable Neo-Hookean rest Hessian and lumped mass in Warp; its `RestElasticity` helper supports graph-captured value updates. The example then supplies these matrices to the library's eigensolver.

The current README GIF uses the supplied [`anvil for export-ftetwild.mesh`](../anvil%20for%20export-ftetwild.mesh), read directly from disk: **94,619 vertices, 508,903 tetrahedra, and 283,857 degrees of freedom**. No mesher is invoked. The longest rest extent is normalized to one metre; the illustrative material uses `E = 100,000 Pa`, `ν = 0.3`, and `ρ = 1,000 kg/m³`, with a free body, stable Neo-Hookean rest Hessian, and lumped mass. Six rigid modes are checked and omitted, then the first 20 elastic modes are shown in order.

Each mode's maximum displacement is **5% of the rest bounding-box diagonal**, checked over **all vertices**, including the interior. The normalized diagonal is 1.18974049 m, so every peak displacement is 0.05948702 m. The smooth, continuous gptoolbox `okloop(256,-4*pi/3,-pi/2)` palette has **no isoline stripes**. Color follows instantaneous displacement with one fixed scale for the entire sequence. Each mode eases from rest to peak and back with `squease`; the camera stays fixed against a white studio background. The GIF has 660 frames at 20 fps, lasting 33 seconds.

The maximum original-problem normalized residual is **2.62e-9**, and the mass-orthogonality error is **1.91e-14**. Frequencies span **1.811–9.166 Hz** under the illustrative material parameters. These are exaggerated linear mode shapes. See the [numerical report](../results/anvil_modes.json), [scale and style metadata](../results/anvil_teaser.json), and [evaluated scene validation](../results/anvil_teaser_validation.json).

Reproduce with Blender 4.5 and [gifski](https://gif.ski/):

```bash
python -m examples.dragon_modes --mesh "anvil for export-ftetwild.mesh" \
  --out results/anvil_modes.npz
blender -b --python examples/render_modes.py -- \
  --input results/anvil_modes.npz --output results/anvil_base.blend \
  --name Anvil --up-axis Y --yaw-degrees 90
blender -b --python examples/render_teaser.py -- \
  --base results/anvil_base.blend --input results/anvil_modes.npz \
  --output results/anvil_teaser.blend --name Anvil --up-axis Y --yaw-degrees 90 \
  --amplitude-bbd 0.05 --frames build/anvil_frames --render
blender -b results/anvil_teaser.blend --python-exit-code 1 --python examples/validate_teaser.py
gifski --fps 20 --quality 85 --width 720 --repeat 0 \
  --output results/anvil_teaser.gif build/anvil_frames/frame_*.png
```

The mesh name, orientation, and displacement cap are rendering options; the eigensolver and Hessian are unchanged. The [`v0.1.3` release](https://github.com/alecjacobson/warpack/releases/tag/v0.1.3) includes the input mesh, modes, editable Blender scene, GIF, and validation metadata.


For the fast pure-Warp solve and optional displacement export, see the
[release recipe](releases/v0.2.0.md) and [Anvil example](../examples/anvil_block.py).
That export omits the boundary-face/rendering metadata used by this rendering pipeline.
