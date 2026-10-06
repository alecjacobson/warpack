"""Show how display amplitude changes the same, verified mode-09 eigenvector."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def main():
    data = np.load("results/dragon_ftetwild_modes.npz")
    x, faces, u = data["vertices"], data["faces"], data["modes"][14]
    magnitude = np.linalg.norm(u, axis=1)
    tip = int(magnitude.argmax())
    scale = json.loads(Path("results/dragon_ftetwild_teaser.json").read_text())[
        "peak_visual_scales"
    ][8]
    scales = [0, 0.025 / magnitude.max(), scale]
    poses = [x + s * u for s in scales]
    low = np.min(poses, axis=(0, 1))
    high = np.max(poses, axis=(0, 1))
    fig = plt.figure(figsize=(15, 5.4), facecolor="white")
    for j, (pose, title) in enumerate(
        zip(
            poses,
            [
                "Rest geometry",
                "Peak displacement: 2.5% of body length",
                "README peak: 76.8% of body length",
            ],
        )
    ):
        ax = fig.add_subplot(1, 3, j + 1, projection="3d")
        scalar = magnitude[faces].mean(axis=1) / magnitude.max()
        collection = Poly3DCollection(
            pose[faces], facecolors=plt.cm.viridis(scalar), linewidths=0, rasterized=True
        )
        ax.add_collection3d(collection)
        ax.scatter(*pose[tip], color="#e64b35", s=24, depthshade=False)
        ax.set_xlim(low[0], high[0])
        ax.set_ylim(low[1], high[1])
        ax.set_zlim(low[2], high[2])
        ax.set_box_aspect(high - low)
        ax.view_init(elev=23, azim=-56)
        ax.set_axis_off()
        ax.set_title(title, fontsize=11)
    fig.suptitle("Mode 09: identical eigenvector, different display amplitudes", fontsize=16)
    fig.text(
        0.5,
        0.06,
        "Fixed camera and bounds. Color shows normalized mode magnitude on the rest mesh; red dot marks the whisker tip.\nNeither deformation is a nonlinear simulation. Reducing display amplitude leaves the eigenproblem unchanged.",
        ha="center",
        fontsize=10,
    )
    fig.subplots_adjust(left=0, right=1, bottom=0.12, top=0.85, wspace=0)
    fig.savefig("results/mode09_amplitude_comparison.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
