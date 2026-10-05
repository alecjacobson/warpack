"""Blender 4.5: editable 20-mode animation with magnitude pseudocolor.

blender -b --python examples/render_modes.py -- --input results/dragon_modes.npz
Add --render to render the full 40 second film; --preview renders one still.
"""

import argparse
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

p = argparse.ArgumentParser()
p.add_argument("--input", default="results/dragon_modes.npz")
p.add_argument("--output", default="results/dragon_modes.blend")
p.add_argument("--render", action="store_true")
p.add_argument("--preview", action="store_true")
p.add_argument("--samples", type=int, default=24)
p.add_argument("--resolution", type=int, default=960)
args = p.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])
d = np.load(args.input)
x = d["vertices"]
faces = d["faces"]
modes = d["modes"][6:26]
vals = d["eigenvalues"][6:26]
ids, remap = np.unique(faces, return_inverse=True)
faces = remap.reshape(-1, 3)
x = x[ids]
modes = modes[:, ids]
# Input Y-up -> Blender Z-up, a proper rotation.
x = x[:, [0, 2, 1]] * np.array([1, -1, 1])
modes = modes[:, :, [0, 2, 1]] * np.array([1, -1, 1])
x[:, 2] -= x[:, 2].min()
x[:, 2] += 0.06
x[:, :2] -= (x[:, :2].min(0) + x[:, :2].max(0)) / 2
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.render.use_persistent_data = True
scene.cycles.samples = args.samples
scene.cycles.use_denoising = True
scene.cycles.denoiser = "OPTIX"
prefs = bpy.context.preferences.addons["cycles"].preferences
prefs.compute_device_type = "OPTIX"
prefs.get_devices()
for dev in prefs.devices:
    dev.use = dev.type == "OPTIX"
scene.cycles.device = "GPU"
scene.render.resolution_x = args.resolution
scene.render.resolution_y = round(args.resolution * 0.75)
scene.render.resolution_percentage = 100
scene.render.fps = 24
scene.frame_start = 1
scene.frame_end = 960
scene.render.film_transparent = True
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
world = bpy.data.worlds.new("White studio")
scene.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)
world.node_tree.nodes["Background"].inputs[1].default_value = 0.35
mesh = bpy.data.meshes.new("Original dragon-H boundary")
mesh.from_pydata(x.tolist(), [], faces.tolist())
mesh.update()
obj = bpy.data.objects.new("Dragon | twenty elastic eigenmodes", mesh)
scene.collection.objects.link(obj)
for poly in mesh.polygons:
    poly.use_smooth = True
obj.shape_key_add(name="Rest")
scales = []
for j, mode in enumerate(modes):
    mag = np.linalg.norm(mode, axis=1)
    # Exactly 2.5% of body length at the largest surface displacement.
    amplitude = 0.025 / mag.max()
    scales.append(float(amplitude))
    key = obj.shape_key_add(
        name=f"Mode {j + 1:02d} | {np.sqrt(max(0, vals[j])) / (2 * np.pi):.3f} Hz"
    )
    key.data.foreach_set("co", (x + amplitude * mode).astype(np.float32).ravel())
    key.slider_min = -1.0
    key.slider_max = 1.0
    fc = key.driver_add("value")
    fc.driver.expression = (
        f"sin(2*pi*(frame-1)/48) if {j * 48 + 1} <= frame < {(j + 1) * 48 + 1} else 0"
    )
    attr = mesh.color_attributes.new(name=f"Magnitude_{j:02d}", type="FLOAT_COLOR", domain="POINT")
    stops = np.array(
        [
            [0.025, 0.06, 0.28],
            [0.02, 0.3, 0.65],
            [0.02, 0.7, 0.75],
            [0.6, 0.85, 0.25],
            [1.0, 0.6, 0.06],
            [0.7, 0.025, 0.035],
        ]
    )
    u = np.clip(mag / mag.max(), 0, 1) * (len(stops) - 1)
    ind = np.minimum(u.astype(int), len(stops) - 2)
    f = u - ind
    rgb = stops[ind] * (1 - f[:, None]) + stops[ind + 1] * f[:, None]
    # Colormap values are sRGB; material color data is linear.
    rgb = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    attr.data.foreach_set(
        "color", np.column_stack([rgb, np.ones(len(rgb))]).astype(np.float32).ravel()
    )
mat = bpy.data.materials.new("Displacement magnitude | per-mode normalized")
mat.use_nodes = True
nodes = mat.node_tree.nodes
links = mat.node_tree.links
bsdf = nodes.get("Principled BSDF")
bsdf.inputs["Roughness"].default_value = 0.4
bsdf.inputs["Specular IOR Level"].default_value = 0.22
prev = None
for j in range(20):
    attr = nodes.new("ShaderNodeVertexColor")
    attr.layer_name = f"Magnitude_{j:02d}"
    if prev is None:
        prev = attr.outputs["Color"]
    else:
        mix = nodes.new("ShaderNodeMixRGB")
        mix.blend_type = "MIX"
        links.new(prev, mix.inputs[1])
        links.new(attr.outputs["Color"], mix.inputs[2])
        fc = mix.inputs[0].driver_add("default_value")
        fc.driver.expression = f"1 if {j * 48 + 1} <= frame < {(j + 1) * 48 + 1} else 0"
        prev = mix.outputs[0]
links.new(prev, bsdf.inputs["Base Color"])
mesh.materials.append(mat)
obj["normalization"] = (
    "M-orthonormal modes; visual amplitude scaled independently to maximum 2.5% body length."
)
obj["boundary"] = "Free body. Six rigid modes omitted."
obj["playback"] = "Each mode plays one cycle over two seconds; physical frequencies are labelled."
obj["visual_scales"] = scales


def aim(o, p):
    o.rotation_euler = (Vector(p) - o.location).to_track_quat("-Z", "Y").to_euler()


def area(name, pos, power, size):
    light = bpy.data.lights.new(name, "AREA")
    light.energy = power
    light.shape = "DISK"
    light.size = size
    ob = bpy.data.objects.new(name, light)
    scene.collection.objects.link(ob)
    ob.location = pos
    aim(ob, (0, 0, 0.25))


area("Key softbox", (-1, -1.5, 2), 100, 1.8)
area("Fill softbox", (1.5, -0.3, 1), 45, 1.5)
area("Rim softbox", (-0.3, 1.5, 1.5), 90, 1.3)
bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, 0))
floor = bpy.context.object
floor.name = "Soft studio shadow"
floor.is_shadow_catcher = True
matfloor = bpy.data.materials.new("White matte floor")
matfloor.diffuse_color = (1, 1, 1, 1)
floor.data.materials.append(matfloor)
camdata = bpy.data.cameras.new("Studio camera")
cam = bpy.data.objects.new("Studio camera", camdata)
scene.collection.objects.link(cam)
scene.camera = cam
cam.location = (0.45, -1.5, 0.7)
target = Vector((0, 0, 0.26))
aim(cam, target)
camdata.type = "ORTHO"
camdata.ortho_scale = 1.3


# Typography lives in the camera plane.
def text_obj(name, body, position, size):
    curve = bpy.data.curves.new(name, "FONT")
    curve.body = body
    curve.size = size
    curve.align_x = "CENTER"
    ob = bpy.data.objects.new(name, curve)
    scene.collection.objects.link(ob)
    ob.parent = cam
    ob.location = position
    material = bpy.data.materials.get("Ink")
    if material is None:
        material = bpy.data.materials.new("Ink")
        material.use_nodes = True
        nn = material.node_tree.nodes
        nn.clear()
        e = nn.new("ShaderNodeEmission")
        e.inputs[0].default_value = (0.025, 0.035, 0.055, 1)
        out = nn.new("ShaderNodeOutputMaterial")
        material.node_tree.links.new(e.outputs[0], out.inputs[0])
    curve.materials.append(material)
    return ob


text_obj("Title", "DRAGON / ELASTIC MODES", (0, 0.41, -1), 0.035)
text_obj(
    "Legend",
    "BLUE  0   /   DISPLACEMENT MAGNITUDE   /   MAX  RED",
    (0, -0.40, -1),
    0.014,
)
text_obj(
    "Scale",
    "Free body  |  2.5% peak displacement  |  2 s per cycle",
    (0, -0.44, -1),
    0.014,
)
for j in range(20):
    freq = np.sqrt(max(0, vals[j])) / (2 * np.pi)
    ob = text_obj(
        f"Label {j + 1:02d}",
        f"MODE {j + 1:02d}   /   {freq:.3f} Hz",
        (0, 0.365, -1),
        0.023,
    )
    fc = ob.driver_add("hide_render")
    fc.driver.expression = f"not ({j * 48 + 1} <= frame < {(j + 1) * 48 + 1})"
    fc = ob.driver_add("hide_viewport")
    fc.driver.expression = f"not ({j * 48 + 1} <= frame < {(j + 1) * 48 + 1})"
scene.use_nodes = True
nt = scene.node_tree
nt.nodes.clear()
rl = nt.nodes.new("CompositorNodeRLayers")
over = nt.nodes.new("CompositorNodeAlphaOver")
over.inputs[1].default_value = (1, 1, 1, 1)
out = nt.nodes.new("CompositorNodeComposite")
nt.links.new(rl.outputs["Image"], over.inputs[2])
nt.links.new(over.outputs[0], out.inputs[0])
outpath = Path(args.output).resolve()
outpath.parent.mkdir(parents=True, exist_ok=True)
scene.frame_set(13)
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = str(outpath.with_suffix(".png"))
bpy.ops.wm.save_as_mainfile(filepath=str(outpath))
if args.preview:
    bpy.ops.render.render(write_still=True)
if args.render:
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "HIGH"
    scene.render.filepath = str(outpath.with_suffix(".mp4"))
    bpy.ops.render.render(animation=True)
print("Saved editable twenty-mode animation:", outpath, flush=True)
