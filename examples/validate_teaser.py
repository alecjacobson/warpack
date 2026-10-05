"""Validate the saved teaser's evaluated geometry, colors, labels, and bounds.

blender -b results/dragon_teaser.blend --python examples/validate_teaser.py
"""

import json
from pathlib import Path

import bpy
import numpy as np

scene = bpy.context.scene
obj = bpy.data.objects["Dragon | twenty elastic eigenmodes"]
mesh = obj.data
report_path = Path(bpy.data.filepath).with_suffix(".json")
report = json.loads(report_path.read_text())
rest = np.empty((len(mesh.vertices), 3), dtype=np.float32)
mesh.shape_keys.key_blocks[0].data.foreach_get("co", rest.ravel())
low, high = np.array(report["allowed_box_low"]), np.array(report["allowed_box_high"])
color_max = report["global_color_max_displacement_m"]
material = mesh.materials[0]
pulse_node = next(n for n in material.node_tree.nodes if n.bl_idname == "ShaderNodeValue")
max_color_error = 0.0
max_pose_error = 0.0
for j in range(20):
    attr = mesh.attributes[f"Peak_displacement_{j:02d}"]
    peak_color = np.empty(len(mesh.vertices), dtype=np.float32)
    attr.data.foreach_get("value", peak_color)
    peak = np.empty_like(rest)
    mesh.shape_keys.key_blocks[j + 1].data.foreach_get("co", peak.ravel())
    last = report["frames_per_mode"] - 1
    for offset in [0, last // 8, last // 2, 3 * last // 4, last]:
        frame = j * report["frames_per_mode"] + offset + 1
        scene.frame_set(frame)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        evaluated = obj.evaluated_get(depsgraph)
        current = np.empty_like(rest)
        evaluated.data.vertices.foreach_get("co", current.ravel())
        value = float(pulse_node.outputs[0].default_value)
        expected = rest + value * (peak - rest)
        max_pose_error = max(max_pose_error, float(np.max(abs(current - expected))))
        actual_color = np.linalg.norm(current - rest, axis=1) / color_max
        max_color_error = max(
            max_color_error, float(np.max(abs(actual_color - value * peak_color)))
        )
        assert np.all(current >= low - 1e-6) and np.all(current <= high + 1e-6)
        visible = [bpy.data.objects[f"Label {m + 1:02d}"].hide_render is False for m in range(20)]
        assert sum(visible) == 1 and visible[j]
        if offset in [0, last]:
            assert value < 1e-12
            assert np.max(abs(current - rest)) < 1e-6
        if offset == last // 2:
            assert abs(value - 1) < 1e-7
assert max_pose_error < 1e-6
assert max_color_error < 1e-6
result = {
    "evaluated_pose_checks": 100,
    "max_pose_interpolation_error_m": max_pose_error,
    "max_instantaneous_color_scalar_error": max_color_error,
    "all_peak_poses_inside_2x_rest_box": True,
    "every_mode_returns_to_rest": True,
    "exactly_one_correct_mode_label_per_checked_frame": True,
    "continuous_bound_argument": "Every shape is a convex interpolation of rest and its bounded peak; baked keyframes use linear interpolation.",
}
Path("results/teaser_validation.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
