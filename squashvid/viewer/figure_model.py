"""Build the 3D figure model the viewer poses: a child from Microsoft's Rocketbox avatar library (MIT), in a
t-shirt and shorts, as one glTF with its bones renamed to the Mixamo names view3d.js poses, plus masks that say
which texture pixels are shirt, shorts and skin, so the viewer can paint each figure's clothes in its player's
colours and its skin in one neutral tone.

    python -I -m squashvid.viewer.figure_model        # needs Blender (/Applications/Blender.app or BLENDER=...)

Downloads Male_Child_02 (FBX and its 2048 px textures) into work/figure_src/, and writes
static/vendor/models/kid.glb, kid_mask_body.png and kid_mask_head.png.

The masks come from the texture colours: on the body, the shirt is the saturated, bright orange, the shorts the dull
grey-green in the top band, and skin the rest that is brown (the sandals are dark and grey); on the head, hair is
the dark top of the texture less the ears (saturated), and the eyes, mouth and teeth (bottom left) are left out of
the skin. Each mask is a greyscale label image (LABEL_STEP x 1 shirt, 2 shorts, 3 skin, 0 left as it is); not RGBA,
as a browser canvas premultiplies alpha and would lose the colour channels.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import urllib.request
from pathlib import Path

import numpy as np

from squashvid.ball.clips import WORK

AVATAR = "Male_Child_02"
CODE = "cm012"
RAW = f"https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Avatars/Children/{AVATAR}"
TEX = 1024
LABEL_STEP = 60
SHIRT, SHORTS, SKIN = 1, 2, 3
OUT = Path(__file__).with_name("static") / "vendor" / "models"

# Rocketbox (3ds Max Biped) bone -> the Mixamo name view3d.js poses.
RENAME = {
    "Pelvis": "Hips", "Spine": "Spine", "Spine1": "Spine1", "Spine2": "Spine2", "Neck": "Neck", "Head": "Head",
    **{f"{s} {b}": f"{side}{m}" for s, side in (("L", "Left"), ("R", "Right")) for b, m in (
        ("Clavicle", "Shoulder"), ("UpperArm", "Arm"), ("Forearm", "ForeArm"), ("Hand", "Hand"), ("Finger2", "HandMiddle1"),
        ("Thigh", "UpLeg"), ("Calf", "Leg"), ("Foot", "Foot"), ("Toe0", "ToeBase"))},
}

BLENDER_SCRIPT = textwrap.dedent('''
    import bpy, sys, json
    src, out, rename, tex = sys.argv[-4], sys.argv[-3], json.loads(sys.argv[-2]), sys.argv[-1]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=src)
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    mesh = next(o for o in bpy.data.objects if o.type == "MESH")
    for o in list(bpy.data.objects):
        if o not in (arm, mesh):
            bpy.data.objects.remove(o)
    arm.animation_data_clear()
    for a in list(bpy.data.actions):
        bpy.data.actions.remove(a)
    for pb in arm.pose.bones:
        pb.matrix_basis.identity()
    prefix = arm.data.bones[0].name.split(" ")[0] + " "
    for b in arm.data.bones:
        short = b.name[len(prefix):] if b.name.startswith(prefix) else b.name
        if short in rename:
            old = b.name
            b.name = rename[short]
            if old in mesh.vertex_groups:
                mesh.vertex_groups[old].name = b.name
    # A bone at the top of the head: the figure is scaled by the trunk, neck to crown.
    top = max((mesh.matrix_world @ v.co).z for v in mesh.data.vertices)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    head = arm.data.edit_bones["Head"]
    end = arm.data.edit_bones.new("HeadTop_End")
    inv = arm.matrix_world.inverted()
    end.head = inv @ (arm.matrix_world @ head.head).copy()
    end.head.z = (inv @ __import__("mathutils").Vector((0, 0, top))).z
    end.tail = end.head.copy(); end.tail.z += 0.05
    end.parent = head
    bpy.ops.object.mode_set(mode="OBJECT")
    for mat in mesh.data.materials:
        part = "body" if mat.name.endswith("body") else "head"
        mat.use_nodes = True
        nt = mat.node_tree; nt.nodes.clear()
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled"); outn = nt.nodes.new("ShaderNodeOutputMaterial")
        bsdf.inputs["Roughness"].default_value = 0.8
        nt.links.new(bsdf.outputs[0], outn.inputs[0])
        col = nt.nodes.new("ShaderNodeTexImage"); col.image = bpy.data.images.load(f"{tex}/{part}_color.jpg")
        nt.links.new(col.outputs["Color"], bsdf.inputs["Base Color"])
        nrm = nt.nodes.new("ShaderNodeTexImage"); nrm.image = bpy.data.images.load(f"{tex}/{part}_normal.jpg")
        nrm.image.colorspace_settings.name = "Non-Color"
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nt.links.new(nrm.outputs["Color"], nm.inputs["Color"]); nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
        mat.name = f"kid_{part}"
    bpy.ops.export_scene.gltf(filepath=out, export_format="GLB", export_animations=False, export_skins=True,
                              export_image_format="JPEG", export_yup=True, use_selection=False)
''')


def _fetch(url: str, dst: Path) -> Path:
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url) as r:
            data = r.read()
        if data.startswith(b"version https://git-lfs"):
            raise RuntimeError(f"{url} is a Git LFS pointer")
        dst.write_bytes(data)
    return dst


def masks(body: np.ndarray, head: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Label masks (uint8, LABEL_STEP x part) for the body and head colour textures (float RGB 0..1, square)."""
    from matplotlib.colors import rgb_to_hsv
    from scipy import ndimage as nd

    def big(m, k):
        lab, n = nd.label(m)
        sz = nd.sum(m, lab, range(1, n + 1))
        return np.isin(lab, 1 + np.flatnonzero(sz > k))

    n = body.shape[0]
    y = np.arange(n)[:, None] / n + np.zeros((1, n))
    x = np.zeros((n, 1)) + np.arange(n)[None, :] / n
    H, used = rgb_to_hsv(body), body.max(axis=2) > 0.03
    shirt = (H[..., 1] > 0.5) & (H[..., 2] > 0.42) & (H[..., 0] < 0.11) & ~((y > 0.8) & ((x < 0.3) | (x > 0.68)))  # the palms (bottom corners) are orange too
    shirt = big(nd.binary_fill_holes(nd.binary_closing(nd.binary_opening(shirt, iterations=2), iterations=6)) & used, n * n // 500)
    shorts = (H[..., 1] < 0.32) & (H[..., 2] > 0.23) & used & ~shirt & (y < 0.3)
    shorts = big(nd.binary_fill_holes(nd.binary_closing(nd.binary_opening(shorts, iterations=3), iterations=8)) & used & ~shirt, n * n // 500)
    skin = used & ~shirt & ~shorts & (H[..., 1] > 0.25) & (H[..., 2] > 0.18)
    skin = nd.binary_opening(nd.binary_closing(skin, iterations=3), iterations=2) & used & ~shirt & ~shorts
    Hh, hused = rgb_to_hsv(head), head.max(axis=2) > 0.03
    hair = big(nd.binary_fill_holes(nd.binary_closing((nd.gaussian_filter(Hh[..., 2], 6) < 0.22) & hused & (y < 0.45), iterations=6)), n * n // 200)
    ears = nd.binary_opening((Hh[..., 1] > 0.55) & (Hh[..., 2] > 0.2) & hused, iterations=2)  # brown skin is saturated, hair not
    face = hused & (~hair | ears) & ~((x < 0.17) & (y > 0.61)) & ~((x < 0.36) & (y > 0.69))  # teeth, mouth and eyes, bottom left
    mb = np.select([shirt, shorts, skin], [SHIRT, SHORTS, SKIN], 0) * LABEL_STEP
    return mb.astype(np.uint8), (face * SKIN * LABEL_STEP).astype(np.uint8)


def build(blender: str | None = None) -> list[Path]:
    from PIL import Image

    src = WORK / "figure_src"
    fbx = _fetch(f"{RAW}/Export/{AVATAR}.fbx", src / f"{AVATAR}.fbx")
    tex = src / "tex"
    tex.mkdir(parents=True, exist_ok=True)
    imgs = {}
    for part in ("body", "head"):
        for kind in ("color", "normal"):
            tga = _fetch(f"{RAW}/Textures/{CODE}_{part}_{kind}.tga", src / f"{CODE}_{part}_{kind}.tga")
            im = Image.open(tga).convert("RGB").resize((TEX, TEX), Image.LANCZOS)
            im.save(tex / f"{part}_{kind}.jpg", quality=88)
            imgs[part, kind] = np.asarray(im, float) / 255
    OUT.mkdir(parents=True, exist_ok=True)
    glb = OUT / "kid.glb"
    blender = blender or os.environ.get("BLENDER") or "/Applications/Blender.app/Contents/MacOS/Blender"
    script = src / "build_kid.py"
    script.write_text(BLENDER_SCRIPT)
    import json

    subprocess.run([blender, "-b", "--python", str(script), "--", str(fbx), str(glb), json.dumps(RENAME), str(tex)], check=True, capture_output=True)
    for part, m in zip(("body", "head"), masks(imgs["body", "color"], imgs["head", "color"])):
        Image.fromarray(m, "L").save(OUT / f"kid_mask_{part}.png", optimize=True)
    return [glb, OUT / "kid_mask_body.png", OUT / "kid_mask_head.png"]


if __name__ == "__main__":
    for p in build(sys.argv[1] if len(sys.argv) > 1 else None):
        print(p, p.stat().st_size)
