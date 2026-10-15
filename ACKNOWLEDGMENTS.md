# Acknowledgments and asset licenses

Warpack's code is licensed under the [MIT license](LICENSE). The anvil asset and its derived showcase visualizations have the separate license described below.

## Anvil model

- **Title:** [Blacksmith's Anvil](https://sketchfab.com/3d-models/blacksmiths-anvil-4b11b76fe069496ebd8ccd98f7d37ade)
- **Creator:** [m4rcu5j4ck50n](https://sketchfab.com/m4rcu5j4ck50n)
- **Source:** Sketchfab
- **License:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/), with [full legal terms](https://creativecommons.org/licenses/by/4.0/legalcode)

The supplied `anvil for export-ftetwild.mesh` is a tetrahedral adaptation of this model. Warpack reads that mesh directly; its showcase does not invoke a mesher. For the showcase, the geometry is normalized to a one-metre longest extent, deformed using computed elastic vibration modes, and rendered with displacement-based pseudocoloring, studio lighting, and a white background. Peak displacement is scaled to 5% of the rest bounding-box diagonal.

This attribution applies to the anvil mesh, `results/anvil_teaser.gif`, the derived anvil renderings, and the anvil geometry and animation in the Blender scenes distributed with the [showcase release](https://github.com/alecjacobson/warpack/releases/tag/v0.1.3). These adapted assets and visualizations are provided under CC BY 4.0. Retain the creator credit, source and license links, and an indication of changes when redistributing them. The original creator does not endorse Warpack or these adaptations.

The creator and license were verified against [Sketchfab's model metadata](https://api.sketchfab.com/v3/models/4b11b76fe069496ebd8ccd98f7d37ade) on October 14, 2026.
