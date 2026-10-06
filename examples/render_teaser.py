"""Build/render the eased README GIF scene with continuous displacement color in Blender 4.5.

Requires the studio scene from render_modes.py (also available in v0.1.0).
Render PNGs, then encode with gifski; see README for the complete commands.
"""

import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from teaser_style import GPT_REVISION, bounded_scales, okloop_jet_linear, pulse  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--base", default="results/dragon_modes.blend")
p.add_argument("--input", default="results/dragon_optimized.npz")
p.add_argument("--output", default="results/dragon_teaser.blend")
p.add_argument("--up-axis", choices=["Y", "Z"], default="Y")
p.add_argument("--name", default="Dragon")
p.add_argument("--yaw-degrees", type=float, default=0)
p.add_argument(
    "--amplitude-bbd",
    type=float,
    help="maximum displacement as a fraction of the rest bounding-box diagonal",
)
p.add_argument(
    "--stripes", action="store_true", help="restore the historical alternating scalar stripes"
)
p.add_argument("--frames", default="build/teaser_frames")
p.add_argument("--render", action="store_true")
p.add_argument("--preview", type=int, nargs="*", default=[])
p.add_argument("--resolution", type=int, default=720)
p.add_argument("--samples", type=int, default=16)
p.add_argument("--frames-per-mode", type=int, default=33)
p.add_argument("--fps", type=int, default=20)
args = p.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])
bpy.ops.wm.open_mainfile(filepath=str(Path(args.base).resolve()))
scene = bpy.context.scene
preferences = bpy.context.preferences.addons["cycles"].preferences
preferences.compute_device_type = "OPTIX"
preferences.get_devices()
for device in preferences.devices:
    device.use = device.type == "OPTIX"
scene.cycles.device = "GPU"
# okloop has a much higher lightness than the original dark-blue palette.
# Reduce the studio energy to preserve its hues under the Standard transform.
for light in bpy.data.lights:
    light.energy *= 0.35
scene.world.node_tree.nodes["Background"].inputs[1].default_value = 0.25
obj = bpy.data.objects[f"{args.name} | twenty elastic eigenmodes"]
mesh = obj.data
raw = np.load(args.input)
ids = np.unique(raw["faces"])
x = raw["vertices"][ids]
u = raw["modes"][6:26, ids]
if args.up_axis == "Y":
    x = x[:, [0, 2, 1]] * [1, -1, 1]
    u = u[:, :, [0, 2, 1]] * [1, -1, 1]
angle = np.deg2rad(args.yaw_degrees)
rotation = np.array(
    [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
)
x = x @ rotation.T
u = u @ rotation.T
x[:, 2] -= x[:, 2].min()
x[:, 2] += 0.06
x[:, :2] -= (x[:, :2].min(0) + x[:, :2].max(0)) / 2
scales, box_low, box_high = bounded_scales(x, u)
bbd = float(np.linalg.norm(np.ptp(raw["vertices"], axis=0)))
# The cap includes interior vertices, even though only the surface is rendered.
all_mode_maxima = np.linalg.norm(raw["modes"][6:26], axis=2).max(axis=1)
if args.amplitude_bbd is not None:
    if not np.isfinite(args.amplitude_bbd) or args.amplitude_bbd <= 0:
        raise ValueError("--amplitude-bbd must be positive and finite")
    scales = np.minimum(scales, args.amplitude_bbd * bbd / all_mode_maxima)
displacements = u * scales[:, None, None]
peak = x[None, :, :] + displacements
assert np.all(peak >= box_low - 1e-12) and np.all(peak <= box_high + 1e-12)
magnitudes = np.linalg.norm(displacements, axis=2)
color_max = float(magnitudes.max())
normalized = magnitudes / color_max
frame_count = args.frames_per_mode * 20
curve = pulse(np.linspace(0, 1, args.frames_per_mode))
assert abs(curve[0]) < 1e-14 and abs(curve[-1]) < 1e-14
assert abs(curve.max() - 1) < 1e-14
mesh.shape_keys.animation_data_clear()
for j in range(20):
    key = mesh.shape_keys.key_blocks[j + 1]
    key.slider_min = 0
    key.data.foreach_set("co", peak[j].astype(np.float32).ravel())
    key.value = 0
    start = j * args.frames_per_mode + 1
    for frame in {1, start - 1, start + args.frames_per_mode, frame_count}:
        key.keyframe_insert(data_path="value", frame=max(1, frame))
    for offset, value in enumerate(curve):
        key.value = float(value)
        key.keyframe_insert(data_path="value", frame=start + offset)
    key.value = 0
for name in [attr.name for attr in mesh.color_attributes]:
    mesh.color_attributes.remove(mesh.color_attributes[name])
for j in range(20):
    name = f"Peak_displacement_{j:02d}"
    if name in mesh.attributes:
        mesh.attributes.remove(mesh.attributes[name])
    attr = mesh.attributes.new(name, "FLOAT", "POINT")
    attr.data.foreach_set("value", normalized[j].astype(np.float32))
mat = bpy.data.materials.new(
    "okloop jet | instantaneous displacement" + (" | stripes" if args.stripes else " | continuous")
)
mat.use_nodes = True
nodes, links = mat.node_tree.nodes, mat.node_tree.links
bsdf = nodes.get("Principled BSDF")
bsdf.inputs["Roughness"].default_value = 0.48
bsdf.inputs["Specular IOR Level"].default_value = 0.18
selected = None
for j in range(20):
    attr = nodes.new("ShaderNodeAttribute")
    attr.attribute_name = f"Peak_displacement_{j:02d}"
    if selected is None:
        selected = attr.outputs["Fac"]
    else:
        mix = nodes.new("ShaderNodeMixRGB")
        links.new(selected, mix.inputs[1])
        links.new(attr.outputs["Fac"], mix.inputs[2])
        fc = mix.inputs[0].driver_add("default_value")
        fc.driver.expression = f"1 if {j * args.frames_per_mode + 1} <= frame < {(j + 1) * args.frames_per_mode + 1} else 0"
        selected = mix.outputs[0]
amplitude = nodes.new("ShaderNodeValue")
amplitude.label = "gptoolbox squease: 0 to peak to 0"
for frame in range(1, frame_count + 1):
    amplitude.outputs[0].default_value = float(curve[(frame - 1) % args.frames_per_mode])
    amplitude.outputs[0].keyframe_insert("default_value", frame=frame)
scalar = nodes.new("ShaderNodeMath")
scalar.operation = "MULTIPLY"
links.new(selected, scalar.inputs[0])
links.new(amplitude.outputs[0], scalar.inputs[1])
# Packed 256-entry linear-RGB lookup of okloop(256,-4*pi/3,-pi/2).
image = bpy.data.images.new("gptoolbox okloop jet 256", width=256, height=1, float_buffer=True)
image.colorspace_settings.name = "Non-Color"
palette = okloop_jet_linear()
image.pixels.foreach_set(np.column_stack([palette, np.ones(256)]).astype(np.float32).ravel())
image.pack()
texture = nodes.new("ShaderNodeTexImage")
texture.image = image
texture.interpolation = "Linear"
texture.extension = "EXTEND"
coord_scale = nodes.new("ShaderNodeMath")
coord_scale.operation = "MULTIPLY_ADD"
coord_scale.inputs[1].default_value = 255 / 256
coord_scale.inputs[2].default_value = 0.5 / 256
links.new(scalar.outputs[0], coord_scale.inputs[0])
coords = nodes.new("ShaderNodeCombineXYZ")
coords.inputs[1].default_value = 0.5
links.new(coord_scale.outputs[0], coords.inputs[0])
links.new(coords.outputs[0], texture.inputs["Vector"])
if args.stripes:
    # Polyscope stripe convention: darken alternating scalar bands by 0.65.
    modulo = nodes.new("ShaderNodeMath")
    modulo.operation = "MODULO"
    modulo.inputs[1].default_value = 0.1
    links.new(scalar.outputs[0], modulo.inputs[0])
    stripe = nodes.new("ShaderNodeMath")
    stripe.operation = "GREATER_THAN"
    stripe.inputs[1].default_value = 0.05
    links.new(modulo.outputs[0], stripe.inputs[0])
    darkness = nodes.new("ShaderNodeMath")
    darkness.operation = "MULTIPLY_ADD"
    darkness.inputs[1].default_value = -0.35
    darkness.inputs[2].default_value = 1
    links.new(stripe.outputs[0], darkness.inputs[0])
    shade = nodes.new("ShaderNodeMixRGB")
    shade.blend_type = "MULTIPLY"
    shade.inputs[0].default_value = 1
    links.new(texture.outputs["Color"], shade.inputs[1])
    links.new(darkness.outputs[0], shade.inputs[2])
    links.new(shade.outputs[0], bsdf.inputs["Base Color"])
else:
    links.new(texture.outputs["Color"], bsdf.inputs["Base Color"])
mesh.materials.clear()
mesh.materials.append(mat)
obj["visual_scales"] = scales.tolist()
obj["normalization"] = (
    f"Maximum displacement capped at {100 * args.amplitude_bbd:g}% of rest bounding-box diagonal; includes interior vertices."
    if args.amplitude_bbd is not None
    else "Positive peak poses fit inside centered 2x rest bounding box, with 5% margin."
)
obj["playback"] = "Modes 1-20 in order; gptoolbox squease from rest to peak and back."
obj["color_max_displacement_m"] = color_max
obj["color_normalization"] = (
    "Fixed global maximum displayed displacement; scalar grows with instantaneous amplitude."
)
# Frame the union of all endpoint poses, which bounds every interpolated pose.
low = np.minimum(x.min(0), peak.min(axis=(0, 1)))
high = np.maximum(x.max(0), peak.max(axis=(0, 1)))
center = (low + high) / 2
cam = scene.camera
view_direction = Vector((0.45, -1.5, 0.44)).normalized()
cam.location = Vector(center) + 3 * view_direction
cam.rotation_euler = (-view_direction).to_track_quat("-Z", "Y").to_euler()
bpy.context.view_layer.update()
rotation = np.array(cam.rotation_euler.to_matrix())
projected_min, projected_max = np.full(3, np.inf), np.full(3, -np.inf)
for pose in [x, *peak]:
    projected = (pose - center) @ rotation
    projected_min = np.minimum(projected_min, projected.min(0))
    projected_max = np.maximum(projected_max, projected.max(0))
projected_span = projected_max - projected_min
# Center the silhouette within the camera's image plane.
center += rotation[:, 0] * (projected_min[0] + projected_max[0]) / 2
center += rotation[:, 1] * (projected_min[1] + projected_max[1]) / 2
cam.location = Vector(center) + 3 * view_direction
cam.data.ortho_scale = float(max(projected_span[0] / 0.90, projected_span[1] / (0.75 * 0.76)))
scale = cam.data.ortho_scale
for name, ypos, font_size in [
    ("Title", 0.325, 0.027),
    ("Legend", -0.328, 0.013),
    ("Scale", -0.352, 0.011),
]:
    ob = bpy.data.objects[name]
    ob.location = (0, ypos * scale, -1)
    ob.data.size = font_size * scale
bpy.data.objects["Legend"].data.body = "0    /    INSTANTANEOUS DISPLACEMENT    /    GLOBAL MAX"
bpy.data.objects["Scale"].data.body = (
    f"okloop jet  |  squease  |  peak displacement: {100 * args.amplitude_bbd:g}% of bounding-box diagonal"
    if args.amplitude_bbd is not None
    else "okloop jet  |  squease  |  peak poses within 2x rest bounds"
)
for j in range(20):
    ob = bpy.data.objects[f"Label {j + 1:02d}"]
    ob.location = (0, 0.290 * scale, -1)
    ob.data.size = 0.018 * scale
    for fc in ob.animation_data.drivers:
        fc.driver.expression = (
            f"not ({j * args.frames_per_mode + 1} <= frame < {(j + 1) * args.frames_per_mode + 1})"
        )
# Compact fixed-scale legend using the same palette and optional stripe bands.
bar_image = bpy.data.images.new(
    "okloop displacement legend", width=256, height=1, float_buffer=True
)
bar_image.colorspace_settings.name = "Non-Color"
bar_shade = (
    np.where(np.mod(np.linspace(0, 1, 256), 0.1) > 0.05, 0.65, 1.0)
    if args.stripes
    else np.ones(256)
)
bar_image.pixels.foreach_set(
    np.column_stack([palette * bar_shade[:, None], np.ones(256)]).astype(np.float32).ravel()
)
bar_image.pack()
bar_mat = bpy.data.materials.new("Fixed displacement color scale")
bar_mat.use_nodes = True
bn, bl = bar_mat.node_tree.nodes, bar_mat.node_tree.links
bn.clear()
bt = bn.new("ShaderNodeTexImage")
bt.image = bar_image
bt.extension = "EXTEND"
bc = bn.new("ShaderNodeTexCoord")
bl.new(bc.outputs["Generated"], bt.inputs["Vector"])
be = bn.new("ShaderNodeEmission")
bl.new(bt.outputs["Color"], be.inputs["Color"])
bo = bn.new("ShaderNodeOutputMaterial")
bl.new(be.outputs[0], bo.inputs["Surface"])
bar_mesh = bpy.data.meshes.new("Legend bar")
bar_mesh.from_pydata(
    [(-0.2, -0.004, 0), (0.2, -0.004, 0), (0.2, 0.004, 0), (-0.2, 0.004, 0)], [], [(0, 1, 2, 3)]
)
bar = bpy.data.objects.new("Displacement colorbar", bar_mesh)
scene.collection.objects.link(bar)
bar.parent = cam
bar.location = (0, -0.300 * scale, -1)
bar.scale = (scale, scale, scale)
bar_mesh.materials.append(bar_mat)
bar.visible_shadow = False
bpy.data.objects["Soft studio shadow"].location.z = float(low[2] - 0.035)
scene.frame_end = frame_count
scene.render.fps = args.fps
scene.render.resolution_x = args.resolution
scene.render.resolution_y = round(args.resolution * 0.75)
scene.cycles.samples = args.samples
# Linear interpolation preserves the bounds between baked samples.
for datablock in [mesh.shape_keys, mat.node_tree]:
    action = datablock.animation_data.action
    for fc in action.fcurves:
        for point in fc.keyframe_points:
            point.interpolation = "LINEAR"
frames = Path(args.frames).resolve()
frames.mkdir(parents=True, exist_ok=True)
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGB"
scene.render.filepath = str(frames / "frame_")
scene.frame_set(1)
output = Path(args.output).resolve()
bpy.ops.wm.save_as_mainfile(filepath=str(output), compress=True)
report = {
    "input": str(args.input),
    "object_name": obj.name,
    "name": args.name,
    "yaw_degrees": args.yaw_degrees,
    "amplitude_bbd_requested": args.amplitude_bbd,
    "rest_bounding_box_diagonal_m": bbd,
    "all_vertex_maximum_displacement_per_mode_m": (all_mode_maxima * scales).tolist(),
    "all_vertex_maximum_displacement_per_mode_bbd": (all_mode_maxima * scales / bbd).tolist(),
    "stripes": args.stripes,
    "up_axis": args.up_axis,
    "gptoolbox_revision": GPT_REVISION,
    "okloop": {"count": 256, "arc": "-4*pi/3", "shift": "-pi/2"},
    "isoline_band_width_normalized": 0.05 if args.stripes else None,
    "isoline_darkness": 0.65 if args.stripes else None,
    "modes": list(range(1, 21)),
    "frames_per_mode": args.frames_per_mode,
    "frames": frame_count,
    "fps": args.fps,
    "resolution": [args.resolution, round(args.resolution * 0.75)],
    "peak_visual_scales": scales.tolist(),
    "maximum_displacement_per_mode_m": magnitudes.max(axis=1).tolist(),
    "global_color_max_displacement_m": color_max,
    "rest_box_low": x.min(0).tolist(),
    "rest_box_high": x.max(0).tolist(),
    "allowed_box_low": box_low.tolist(),
    "allowed_box_high": box_high.tolist(),
    "animation_box_low": low.tolist(),
    "animation_box_high": high.tolist(),
    "maximum_extent_ratio": float(((high - low) / np.ptp(x, axis=0)).max()),
    "all_continuous_poses_within_2x_box": True,
    "pulse": curve.tolist(),
    "physical_note": "Exaggerated linear modal displacements, not a nonlinear deformation simulation.",
}
output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
for frame in args.preview:
    scene.frame_set(frame)
    scene.render.filepath = str(frames / f"preview_{frame:04d}.png")
    bpy.ops.render.render(write_still=True)
scene.render.filepath = str(frames / "frame_")
if args.render:
    bpy.ops.render.render(animation=True)
