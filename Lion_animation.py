"""
KING OF THE FOREST  -  cinematic 3D lion animation (Blender Python)
=====================================================================
A majestic procedural lion (fur shader, flowing mane, whiskers, blinking eyes, IK walk cycle)
walks through a sunlit forest. Camera dollies in; at the end the lion stops and looks straight
into the lens. 7 s | 1080x1920 (9:16) | 30 FPS | MP4 (H.264)

HOW TO USE
  1. Open Blender (4.2 LTS or newer recommended; 3.6+ is supported).
  2. Scripting workspace -> Open -> choose this file -> Run Script (Alt+P). Building takes a few seconds.
  3. Press Space over the 3D viewport to preview the animation.
  4. Render > Render Animation (Ctrl+F12). The MP4 is written next to your .blend file
     (or into ~/Blender_Output if the .blend has not been saved).
  Headless:  blender -b -P 01_King_of_the_Forest.py     (builds and renders by itself)

SETTINGS you may want to change (just below the imports, and USE_VOLUMETRICS further down):
  RENDER_ENGINE = "EEVEE" (fast) or "CYCLES" (slower, more realistic)
  SAMPLES, RESOLUTION_PERCENT (50 = quick preview render), USE_VOLUMETRICS (god-ray haze)
"""
import bpy
import bmesh
import math
import os
import random
import traceback

# ============================================================================
#  USER SETTINGS
# ============================================================================
RENDER_ENGINE = "EEVEE"      # "EEVEE" = fast  |  "CYCLES" = slower, more realistic
SAMPLES = 64                 # anti-aliasing samples per frame
RESOLUTION_PERCENT = 100     # 50 = quick preview render
MOTION_BLUR = True
RENDER_ON_RUN = False        # True = start rendering as soon as the script ends.
                             # (Always renders when run headless: blender -b -P script.py)

FPS = 30
DURATION = 7
F_START = 1
F_END = FPS * DURATION       # 210 frames = exactly 7.0 seconds
RES_X, RES_Y = 1080, 1920    # vertical 9:16


# ============================================================================
#  SMALL UTILITIES
# ============================================================================
def warn(msg):
    print("[warning] " + str(msg))


def safe(fn):
    """Decorator: a cosmetic/optional step that fails must never stop the script."""
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            warn("optional step '%s' was skipped because of an error:" % fn.__name__)
            traceback.print_exc()
            return None
    wrapper.__name__ = fn.__name__
    return wrapper


def try_set(obj, name, value):
    try:
        setattr(obj, name, value)
        return True
    except Exception:
        return False


def lerp(a, b, t):
    return a + (b - a) * t


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def smoothstep(a, b, x):
    t = clamp((x - a) / (b - a))
    return t * t * (3.0 - 2.0 * t)


def rgba(c, a=1.0):
    return (c[0], c[1], c[2], c[3] if len(c) > 3 else a)


def vadd(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vsub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vmul(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def vdot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vcross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def vlen(a):
    return math.sqrt(vdot(a, a))


def vnorm(a):
    l = vlen(a)
    if l < 1e-9:
        return (0.0, 0.0, 1.0)
    return (a[0] / l, a[1] / l, a[2] / l)


def vlerp(a, b, t):
    return (lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t))


def rot_xyz(v, e):
    """Rotate vector v by a Blender 'XYZ' Euler (X first, then Y, then Z)."""
    x, y, z = v
    c, s = math.cos(e[0]), math.sin(e[0])
    y, z = y * c - z * s, y * s + z * c
    c, s = math.cos(e[1]), math.sin(e[1])
    x, z = x * c + z * s, -x * s + z * c
    c, s = math.cos(e[2]), math.sin(e[2])
    x, y = x * c - y * s, x * s + y * c
    return (x, y, z)


def aim_rotation(d):
    """Euler that makes an object's -Z axis (lights, spots) point along direction d."""
    d = vnorm(d)
    return (math.acos(clamp(-d[2], -1.0, 1.0)), 0.0, math.atan2(-d[0], d[1]))


def heading_to(dx, dy):
    """Z rotation so that local +Y faces world direction (dx, dy); valid for dy < 0 (facing -Y = pi)."""
    return math.pi + math.atan2(dx, -dy)


def wrap_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def catmull(p0, p1, p2, p3, t):
    return tuple(0.5 * ((2 * b) + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t * t +
                        (-a + 3 * b - 3 * c + d) * t * t * t)
                 for a, b, c, d in zip(p0, p1, p2, p3))


def spline(points, n):
    """Catmull-Rom resample of a list of tuples into n samples."""
    pts = [points[0]] + list(points) + [points[-1]]
    segs = len(points) - 1
    out = []
    for i in range(n):
        u = i / float(n - 1) * segs
        k = min(int(u), segs - 1)
        out.append(catmull(pts[k], pts[k + 1], pts[k + 2], pts[k + 3], u - k))
    return out


# ---- tiny value-noise (no dependencies) ------------------------------------
def _h(ix, iy, iz, seed):
    n = (ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791) ^ (seed * 2654435761)
    n &= 0xFFFFFFFF
    n = (((n >> 16) ^ n) * 0x45d9f3b) & 0xFFFFFFFF
    n = (((n >> 16) ^ n) * 0x45d9f3b) & 0xFFFFFFFF
    n = (n >> 16) ^ n
    return (n & 0xFFFF) / 65535.0


def vnoise(x, y, z=0.0, seed=0):
    ix, iy, iz = int(math.floor(x)), int(math.floor(y)), int(math.floor(z))
    fx, fy, fz = x - ix, y - iy, z - iz
    fx, fy, fz = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy), fz * fz * (3 - 2 * fz)

    def c(dx, dy, dz):
        return _h(ix + dx, iy + dy, iz + dz, seed)
    x00 = lerp(c(0, 0, 0), c(1, 0, 0), fx)
    x10 = lerp(c(0, 1, 0), c(1, 1, 0), fx)
    x01 = lerp(c(0, 0, 1), c(1, 0, 1), fx)
    x11 = lerp(c(0, 1, 1), c(1, 1, 1), fx)
    return lerp(lerp(x00, x10, fy), lerp(x01, x11, fy), fz)


def fbm(x, y, z=0.0, octaves=4, seed=0):
    total, amp, freq, norm = 0.0, 1.0, 1.0, 0.0
    for _ in range(octaves):
        total += vnoise(x * freq, y * freq, z * freq, seed) * amp
        norm += amp
        amp *= 0.5
        freq *= 2.0
    return total / norm


# ============================================================================
#  GEOMETRY  (pure python lists -> real meshes, no bpy.ops needed)
# ============================================================================
def sphere_geo(segs=18, rings=10):
    verts = [(0.0, 0.0, 1.0)]
    uvs = [(0.5, 1.0)]
    for r in range(1, rings):
        th = math.pi * r / rings
        st, ct = math.sin(th), math.cos(th)
        for s in range(segs):
            ph = 2.0 * math.pi * s / segs
            verts.append((st * math.cos(ph), st * math.sin(ph), ct))
            uvs.append((s / float(segs), 1.0 - r / float(rings)))
    verts.append((0.0, 0.0, -1.0))
    uvs.append((0.5, 0.0))
    bottom = len(verts) - 1

    def idx(r, s):
        return 1 + (r - 1) * segs + (s % segs)
    faces = []
    for s in range(segs):
        faces.append((0, idx(1, s), idx(1, s + 1)))
    for r in range(1, rings - 1):
        for s in range(segs):
            faces.append((idx(r, s), idx(r + 1, s), idx(r + 1, s + 1), idx(r, s + 1)))
    for s in range(segs):
        faces.append((idx(rings - 1, s + 1), idx(rings - 1, s), bottom))
    return verts, faces, uvs


def tube_geo(points, radii, sides=8, cap_start=True, cap_end=True, flat=1.0, up=(0.0, 0.0, 1.0)):
    """Tapered tube along a polyline. Returns (verts, faces, uvs). u = along length, v = around."""
    n = len(points)
    tans = []
    for i in range(n):
        a = points[max(i - 1, 0)]
        b = points[min(i + 1, n - 1)]
        tans.append(vnorm(vsub(b, a)))
    t0 = tans[0]
    ref = up if abs(vdot(t0, up)) < 0.95 else (1.0, 0.0, 0.0)
    nrm = vnorm(vsub(ref, vmul(t0, vdot(ref, t0))))
    verts, uvs, faces = [], [], []
    for i in range(n):
        t = tans[i]
        nrm = vsub(nrm, vmul(t, vdot(nrm, t)))
        nrm = vnorm(nrm) if vlen(nrm) > 1e-6 else (1.0, 0.0, 0.0)
        bn = vcross(t, nrm)
        for j in range(sides):
            a = 2.0 * math.pi * j / sides
            off = vadd(vmul(nrm, math.cos(a) * radii[i]), vmul(bn, math.sin(a) * radii[i] * flat))
            verts.append(vadd(points[i], off))
            uvs.append((i / float(max(n - 1, 1)), j / float(sides)))
    for i in range(n - 1):
        for j in range(sides):
            a = i * sides + j
            b = i * sides + (j + 1) % sides
            faces.append((a, b, b + sides, a + sides))
    if cap_start:
        c = len(verts)
        verts.append(points[0])
        uvs.append((0.0, 0.0))
        for j in range(sides):
            faces.append((c, (j + 1) % sides, j))
    if cap_end:
        c = len(verts)
        verts.append(points[-1])
        uvs.append((1.0, 0.0))
        o = (n - 1) * sides
        for j in range(sides):
            faces.append((c, o + j, o + (j + 1) % sides))
    return verts, faces, uvs


def ribbon_geo(points, widths, side_hint=(0.0, 0.0, 1.0)):
    """Flat strip (grass blade / leaf / light shaft). u = along length, v = across."""
    n = len(points)
    verts, uvs, faces = [], [], []
    for i, p in enumerate(points):
        t = vnorm(vsub(points[min(i + 1, n - 1)], points[max(i - 1, 0)]))
        s = vcross(t, side_hint)
        if vlen(s) < 1e-6:
            s = vcross(t, (1.0, 0.0, 0.0))
        s = vnorm(s)
        w = widths[i] * 0.5
        verts.append(vadd(p, vmul(s, -w)))
        verts.append(vadd(p, vmul(s, w)))
        u = i / float(max(n - 1, 1))
        uvs.append((u, 0.0))
        uvs.append((u, 1.0))
    for i in range(n - 1):
        a = 2 * i
        faces.append((a, a + 1, a + 3, a + 2))
    return verts, faces, uvs


def box_geo(c, s, rot=(0.0, 0.0, 0.0)):
    hx, hy, hz = s[0] / 2.0, s[1] / 2.0, s[2] / 2.0
    local = [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz),
             (-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)]
    verts = [vadd(c, rot_xyz(p, rot)) for p in local]
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return verts, faces


class MeshBuilder(object):
    """Collects many small shapes and turns them into ONE mesh object (fast to render)."""

    def __init__(self):
        self.v, self.f, self.uv = [], [], []

    def add(self, verts, faces, uvs=None):
        o = len(self.v)
        self.v.extend(verts)
        self.f.extend([tuple(i + o for i in face) for face in faces])
        if uvs is None:
            uvs = [(0.0, 0.0)] * len(verts)
        self.uv.extend(uvs)

    def add_box(self, c, s, rot=(0.0, 0.0, 0.0)):
        v, f = box_geo(c, s, rot)
        self.add(v, f)

    def build(self, name, mat=None, smooth=True, parent=None, recalc=True):
        if not self.v:
            return None
        return mesh_object(name, self.v, self.f, self.uv, mat, smooth, parent, recalc=recalc)


def _recalc(me):
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.to_mesh(me)
    bm.free()


def link_obj(ob, parent=None):
    bpy.context.scene.collection.objects.link(ob)
    if parent is not None:
        ob.parent = parent
    return ob


def mesh_object(name, verts, faces, uvs=None, mat=None, smooth=True, parent=None,
                loc=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), recalc=True):
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    if recalc:
        try:
            _recalc(me)
        except Exception:
            pass
    if smooth and len(me.polygons):
        me.polygons.foreach_set('use_smooth', [True] * len(me.polygons))
    if uvs is not None:
        try:
            layer = me.uv_layers.new(name="UVMap")
            vi = [0] * len(me.loops)
            me.loops.foreach_get('vertex_index', vi)
            flat = []
            for i in vi:
                flat.extend(uvs[i])
            layer.data.foreach_set('uv', flat)
        except Exception:
            traceback.print_exc()
    ob = bpy.data.objects.new(name, me)
    ob.location = loc
    ob.rotation_euler = rot
    ob.scale = scale
    link_obj(ob, parent)
    if mat is not None:
        me.materials.append(mat)
    return ob


def blob(name, center=(0, 0, 0), radii=(1, 1, 1), rot=(0, 0, 0), mat=None, parent=None,
         segs=18, rings=10, smooth=True):
    v, f, uv = sphere_geo(segs, rings)
    return mesh_object(name, v, f, uv, mat, smooth, parent, loc=center, rot=rot, scale=radii)


def empty(name, loc=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0), parent=None, size=0.1, kind='PLAIN_AXES'):
    ob = bpy.data.objects.new(name, None)
    ob.empty_display_type = kind
    ob.empty_display_size = size
    ob.location = loc
    ob.rotation_euler = rot
    return link_obj(ob, parent)


def key(ob, path, frame, value=None):
    if value is not None:
        setattr(ob, path, value)
    ob.keyframe_insert(data_path=path, frame=frame)


def hide_from_rays(ob, shadow=True, reflections=True):
    """Keep glow cards / helpers from casting shadows or showing up in reflections."""
    if shadow:
        try_set(ob, 'visible_shadow', False)
    if reflections:
        for a in ('visible_glossy', 'visible_diffuse', 'visible_transmission', 'visible_volume_scatter'):
            try_set(ob, a, False)


def clear_scene():
    try:
        if bpy.context.object and bpy.context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
    except Exception:
        pass
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    for cname in ('meshes', 'materials', 'lights', 'cameras', 'curves', 'particles',
                  'actions', 'textures', 'worlds'):
        coll = getattr(bpy.data, cname, None)
        if coll is None:
            continue
        for item in list(coll):
            try:
                coll.remove(item)
            except Exception:
                pass


# ============================================================================
#  MATERIAL / NODE HELPERS
# ============================================================================
def new_material(name, color=None):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    if color is not None:
        try_set(m, 'diffuse_color', rgba(color))     # viewport colour (also used by my preview tests)
    return m, nt


def node(nt, kind, loc=(0, 0), **props):
    n = nt.nodes.new(kind)
    n.location = loc
    for k, v in props.items():
        setattr(n, k, v)
    return n


def link(nt, a, b):
    nt.links.new(a, b)


def set_in(n, names, value):
    """Set a node input by trying several socket names (Blender renamed many in 4.0)."""
    if isinstance(names, str):
        names = (names,)
    for nm in names:
        if nm in n.inputs:
            try:
                n.inputs[nm].default_value = value
                return True
            except Exception:
                continue
    return False


def mix_rgb(nt, blend='MIX', loc=(0, 0)):
    """Colour mix node that works on every Blender version. Returns (node, fac, a, b, out)."""
    try:
        n = nt.nodes.new('ShaderNodeMixRGB')
        n.location = loc
        n.blend_type = blend
        return n, n.inputs['Fac'], n.inputs['Color1'], n.inputs['Color2'], n.outputs['Color']
    except Exception:
        n = nt.nodes.new('ShaderNodeMix')
        n.location = loc
        n.data_type = 'RGBA'
        n.blend_type = blend
        return n, n.inputs[0], n.inputs[6], n.inputs[7], n.outputs[2]


def color_ramp(nt, stops, loc=(0, 0), fac=None):
    """Colour ramp from [(position, colour), ...] (positions ascending).
    The two default elements become the first/last stop; the rest are created at their final
    position and coloured straight away, so Blender's re-sorting can never scramble them."""
    n = node(nt, 'ShaderNodeValToRGB', loc)
    els = n.color_ramp.elements
    els[0].position = stops[0][0]
    els[0].color = rgba(stops[0][1])
    els[1].position = stops[-1][0]
    els[1].color = rgba(stops[-1][1])
    for pos, col in stops[1:-1]:
        e = els.new(pos)
        e.color = rgba(col)
    if fac is not None:
        link(nt, fac, n.inputs['Fac'])
    return n


def math_node(nt, op, loc=(0, 0), a=None, b=None, clamp_out=False):
    n = node(nt, 'ShaderNodeMath', loc, operation=op)
    n.use_clamp = clamp_out
    for i, v in ((0, a), (1, b)):
        if v is None:
            continue
        if isinstance(v, (int, float)):
            n.inputs[i].default_value = v
        else:
            link(nt, v, n.inputs[i])
    return n


def set_blend(m, additive_shadow=False):
    """Alpha-blended material that does not cast shadows (works on 3.x - 5.x)."""
    try_set(m, 'surface_render_method', 'BLENDED')
    try_set(m, 'blend_method', 'BLEND')
    try_set(m, 'shadow_method', 'NONE')
    try_set(m, 'use_backface_culling', False)
    try_set(m, 'show_transparent_back', False)


def pbr(name, color=(0.5, 0.5, 0.5), rough=0.5, metal=0.0, emit=None, emit_strength=0.0,
        alpha=1.0, coat=0.0, spec=None):
    m, nt = new_material(name, color)
    out = node(nt, 'ShaderNodeOutputMaterial', (400, 0))
    b = node(nt, 'ShaderNodeBsdfPrincipled', (0, 0))
    link(nt, b.outputs['BSDF'], out.inputs['Surface'])
    set_in(b, 'Base Color', rgba(color))
    set_in(b, 'Roughness', rough)
    set_in(b, 'Metallic', metal)
    if spec is not None:
        set_in(b, ('Specular IOR Level', 'Specular'), spec)
    if coat > 0:
        set_in(b, ('Coat Weight', 'Clearcoat'), coat)
    if emit is not None:
        set_in(b, ('Emission Color', 'Emission'), rgba(emit))
        set_in(b, 'Emission Strength', emit_strength)
    if alpha < 1.0:
        set_in(b, 'Alpha', alpha)
        set_blend(m)
    return m


def emission_mat(name, color, strength=5.0):
    m, nt = new_material(name, color)
    out = node(nt, 'ShaderNodeOutputMaterial', (400, 0))
    e = node(nt, 'ShaderNodeEmission', (0, 0))
    set_in(e, 'Color', rgba(color))
    set_in(e, 'Strength', strength)
    link(nt, e.outputs['Emission'], out.inputs['Surface'])
    return m


def glow_card_material(name, color, strength=3.0, power=2.0):
    """Soft radial glow for camera-facing halo planes (UV 0..1, centre = brightest)."""
    m, nt = new_material(name, color)
    out = node(nt, 'ShaderNodeOutputMaterial', (900, 0))
    tc = node(nt, 'ShaderNodeTexCoord', (-700, 0))
    mp = node(nt, 'ShaderNodeMapping', (-500, 0))
    mp.inputs['Scale'].default_value = (2.0, 2.0, 2.0)
    mp.inputs['Location'].default_value = (-1.0, -1.0, 0.0)
    link(nt, tc.outputs['UV'], mp.inputs['Vector'])
    gr = node(nt, 'ShaderNodeTexGradient', (-300, 0), gradient_type='SPHERICAL')
    link(nt, mp.outputs['Vector'], gr.inputs['Vector'])
    pw = math_node(nt, 'POWER', (-100, 0), a=gr.outputs['Fac'], b=power, clamp_out=True)
    em = node(nt, 'ShaderNodeEmission', (100, -100))
    set_in(em, 'Color', rgba(color))
    set_in(em, 'Strength', strength)
    tr = node(nt, 'ShaderNodeBsdfTransparent', (100, 100))
    mx = node(nt, 'ShaderNodeMixShader', (500, 0))
    link(nt, pw.outputs[0], mx.inputs[0])
    link(nt, tr.outputs['BSDF'], mx.inputs[1])
    link(nt, em.outputs['Emission'], mx.inputs[2])
    link(nt, mx.outputs['Shader'], out.inputs['Surface'])
    set_blend(m)
    return m


def billboard(name, size, mat, target, parent=None, loc=(0, 0, 0)):
    """Plane that always faces the camera (Track-To constraint)."""
    h = size / 2.0
    ob = mesh_object(name, [(-h, -h, 0), (h, -h, 0), (h, h, 0), (-h, h, 0)], [(0, 1, 2, 3)],
                     [(0, 0), (1, 0), (1, 1), (0, 1)], mat, False, parent, loc=loc, recalc=False)
    c = ob.constraints.new('TRACK_TO')
    c.target = target
    c.track_axis = 'TRACK_Z'
    c.up_axis = 'UP_Y'
    hide_from_rays(ob)
    return ob


# ============================================================================
#  LIGHTS / CAMERA / WORLD
# ============================================================================
def add_light(kind, name, loc, energy, color=(1.0, 1.0, 1.0), size=None, direction=None,
              spot=None, parent=None):
    ld = bpy.data.lights.new(name, kind)
    ld.energy = energy
    ld.color = (color[0], color[1], color[2])
    if size is not None:
        if kind == 'SUN':
            ld.angle = size
        elif kind == 'AREA':
            ld.size = size
        else:
            ld.shadow_soft_size = size
    if kind == 'SPOT' and spot is not None:
        ld.spot_size = spot
        ld.spot_blend = 0.6
    ob = bpy.data.objects.new(name, ld)
    ob.location = loc
    if direction is not None:
        ob.rotation_euler = aim_rotation(direction)
    link_obj(ob, parent)
    return ob


def add_camera(lens, loc, aim, focus=None, fstop=2.8, clip_end=400.0):
    cd = bpy.data.cameras.new("Camera")
    cd.lens = lens
    cd.sensor_width = 36.0
    cd.clip_start = 0.05
    cd.clip_end = clip_end
    if focus is not None:
        cd.dof.use_dof = True
        cd.dof.focus_object = focus
        cd.dof.aperture_fstop = fstop
    cam = bpy.data.objects.new("Camera", cd)
    link_obj(cam)
    cam.location = loc
    c = cam.constraints.new('TRACK_TO')
    c.target = aim
    c.track_axis = 'TRACK_NEGATIVE_Z'
    c.up_axis = 'UP_Y'
    bpy.context.scene.camera = cam
    return cam


def make_sky(zenith, horizon, ground, sun_dir=None, sun_color=(1.0, 0.8, 0.5), glow=0.0,
             glow_power=12.0, disc=0.0, stars=0.0, strength=1.0, band=0.5):
    """Procedural sky: gradient + optional sun/moon glow + disc + star field.
    sun_dir = direction the light TRAVELS (same vector used for the Sun lamp)."""
    scene = bpy.context.scene
    w = bpy.data.worlds.new("Sky")
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    out = node(nt, 'ShaderNodeOutputWorld', (1700, 0))
    bg = node(nt, 'ShaderNodeBackground', (1500, 0))
    tc = node(nt, 'ShaderNodeTexCoord', (-1500, 0))
    nrm = node(nt, 'ShaderNodeVectorMath', (-1300, 0), operation='NORMALIZE')
    link(nt, tc.outputs['Generated'], nrm.inputs[0])
    sep = node(nt, 'ShaderNodeSeparateXYZ', (-1100, 0))
    link(nt, nrm.outputs['Vector'], sep.inputs['Vector'])
    z = sep.outputs['Z']
    mid = vlerp(horizon, zenith, 0.55)
    ramp = color_ramp(nt, [(0.0, horizon), (band, mid), (1.0, zenith)], (-900, 250), fac=z)
    lt = math_node(nt, 'LESS_THAN', (-900, -50), a=z, b=0.0)
    _, f0, a0, b0, o0 = mix_rgb(nt, 'MIX', (-600, 150))
    link(nt, lt.outputs[0], f0)
    link(nt, ramp.outputs['Color'], a0)
    b0.default_value = rgba(ground)
    cur = o0
    xloc = -300

    def add_layer(fac_socket, color, strength_mul):
        nonlocal cur, xloc
        _, f, a, b, o = mix_rgb(nt, 'ADD', (xloc, 150))
        xloc += 250
        link(nt, cur, a)
        link(nt, fac_socket, f)
        b.default_value = rgba((color[0] * strength_mul, color[1] * strength_mul, color[2] * strength_mul))
        cur = o

    if sun_dir is not None and (glow > 0 or disc > 0):
        to_sun = (-sun_dir[0], -sun_dir[1], -sun_dir[2])
        to_sun = vnorm(to_sun)
        dot = node(nt, 'ShaderNodeVectorMath', (-900, -350), operation='DOT_PRODUCT')
        link(nt, nrm.outputs['Vector'], dot.inputs[0])
        dot.inputs[1].default_value = to_sun
        if glow > 0:
            mx = math_node(nt, 'MAXIMUM', (-700, -350), a=dot.outputs['Value'], b=0.0)
            pw = math_node(nt, 'POWER', (-500, -350), a=mx.outputs[0], b=glow_power, clamp_out=True)
            add_layer(pw.outputs[0], sun_color, glow)
        if disc > 0:
            sb = math_node(nt, 'SUBTRACT', (-700, -550), a=dot.outputs['Value'], b=0.9996)
            ml = math_node(nt, 'MULTIPLY', (-500, -550), a=sb.outputs[0], b=4000.0, clamp_out=True)
            add_layer(ml.outputs[0], sun_color, disc)
    if stars > 0:
        up = math_node(nt, 'MULTIPLY', (-900, -750), a=z, b=6.0, clamp_out=True)
        for scale, thr, mul in ((86.0, 0.045, 1.0), (40.0, 0.042, 2.0)):
            vo = node(nt, 'ShaderNodeTexVoronoi', (-900, -900))
            vo.inputs['Scale'].default_value = scale
            link(nt, nrm.outputs['Vector'], vo.inputs['Vector'])
            sb = math_node(nt, 'SUBTRACT', (-700, -900), a=thr, b=vo.outputs['Distance'])
            ml = math_node(nt, 'MULTIPLY', (-500, -900), a=sb.outputs[0], b=30.0, clamp_out=True)
            up2 = math_node(nt, 'MULTIPLY', (-300, -900), a=ml.outputs[0], b=up.outputs[0], clamp_out=True)
            add_layer(up2.outputs[0], (1.0, 1.0, 1.0), stars * mul)
    link(nt, cur, bg.inputs['Color'])
    bg.inputs['Strength'].default_value = strength
    link(nt, bg.outputs['Background'], out.inputs['Surface'])
    scene.world = w
    return w


def add_volume_box(center, size, density, color=(1, 1, 1), aniso=0.0, name="Haze"):
    v, f = box_geo(center, size)
    ob = mesh_object(name, v, f, None, None, False, recalc=False)
    m, nt = new_material(name + "Mat")
    out = node(nt, 'ShaderNodeOutputMaterial', (300, 0))
    vs = node(nt, 'ShaderNodeVolumeScatter', (0, 0))
    set_in(vs, 'Color', rgba(color))
    set_in(vs, 'Density', density)
    set_in(vs, 'Anisotropy', aniso)
    link(nt, vs.outputs['Volume'], out.inputs['Volume'])
    ob.data.materials.append(m)
    ob.display_type = 'WIRE'
    hide_from_rays(ob)
    return ob


@safe
def add_glare(scene, threshold=0.9, size=8, mix=-0.3):
    """Optional compositor bloom. Skipped silently if this Blender version differs."""
    scene.use_nodes = True
    tree = scene.node_tree
    if tree is None:
        raise RuntimeError("no compositor tree")
    try:
        for n in list(tree.nodes):
            tree.nodes.remove(n)
        rl = tree.nodes.new('CompositorNodeRLayers')
        gl = tree.nodes.new('CompositorNodeGlare')
        co = tree.nodes.new('CompositorNodeComposite')
        gl.glare_type = 'FOG_GLOW'
        gl.threshold = threshold
        gl.size = size
        gl.mix = mix
        try_set(gl, 'quality', 'HIGH')
        tree.links.new(rl.outputs['Image'], gl.inputs['Image'])
        tree.links.new(gl.outputs['Image'], co.inputs['Image'])
    except Exception:
        scene.use_nodes = False
        raise


# ============================================================================
#  RENDER SETTINGS + MP4 EXPORT
# ============================================================================
def output_dir():
    blend = bpy.data.filepath
    if isinstance(blend, str) and blend:
        d = os.path.dirname(blend)
    else:
        d = os.path.join(os.path.expanduser("~"), "Blender_Output")
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    return d


def setup_render(scene, filename, exposure=0.0, volumetrics=False):
    r = scene.render
    r.resolution_x = RES_X
    r.resolution_y = RES_Y
    r.resolution_percentage = RESOLUTION_PERCENT
    r.pixel_aspect_x = 1.0
    r.pixel_aspect_y = 1.0
    r.fps = FPS
    r.fps_base = 1.0
    scene.frame_start = F_START
    scene.frame_end = F_END
    scene.frame_step = 1

    engine = None
    cands = ("CYCLES",) if RENDER_ENGINE.upper() == "CYCLES" else ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE")
    for cand in cands:
        if try_set(r, 'engine', cand):
            engine = cand
            break
    if engine is None:
        warn("Could not select the requested render engine; keeping the default.")
    elif engine == "CYCLES":
        cy = scene.cycles
        cy.samples = SAMPLES
        try_set(cy, 'use_denoising', True)
        try_set(cy, 'max_bounces', 6)
    else:
        ee = scene.eevee
        try_set(ee, 'taa_render_samples', SAMPLES)
        try_set(ee, 'taa_samples', 16)
        # EEVEE legacy (<= 4.1)
        for k, v in (('use_gtao', True), ('gtao_distance', 0.8), ('use_ssr', True),
                     ('use_ssr_refraction', True), ('use_soft_shadows', True),
                     ('use_bloom', True), ('bloom_threshold', 0.9), ('bloom_intensity', 0.12),
                     ('bloom_radius', 6.0), ('shadow_cube_size', '2048'), ('shadow_cascade_size', '2048'),
                     ('use_shadow_high_bitdepth', True)):
            try_set(ee, k, v)
        # EEVEE Next (4.2+)
        for k, v in (('use_raytracing', True), ('ray_tracing_method', 'SCREEN'), ('use_shadows', True)):
            try_set(ee, k, v)
        if volumetrics:
            for k, v in (('volumetric_tile_size', '4'), ('volumetric_samples', 96),
                         ('volumetric_start', 0.1), ('volumetric_end', 70.0),
                         ('use_volumetric_shadows', True), ('volumetric_shadow_samples', 24),
                         ('volumetric_light_clamp', 0.0)):
                try_set(ee, k, v)
        if MOTION_BLUR:
            try_set(ee, 'use_motion_blur', True)
            try_set(ee, 'motion_blur_shutter', 0.4)
    if MOTION_BLUR:
        try_set(r, 'use_motion_blur', True)
        try_set(r, 'motion_blur_shutter', 0.4)
    r.film_transparent = False

    vs = scene.view_settings
    for vt in ("AgX", "Filmic", "Standard"):
        if try_set(vs, 'view_transform', vt):
            break
    try_set(vs, 'exposure', exposure)

    if not filename.lower().endswith('.mp4'):
        filename += '.mp4'
    path = os.path.join(output_dir(), filename)
    r.filepath = path
    try_set(r, 'use_file_extension', True)
    try_set(r.image_settings, 'media_type', 'VIDEO')       # Blender 5.x
    try:
        r.image_settings.file_format = 'FFMPEG'
    except Exception:
        warn("Could not switch the output format to FFMPEG (MP4).")
    try_set(r.image_settings, 'color_mode', 'RGB')
    ff = r.ffmpeg
    try_set(ff, 'format', 'MPEG4')
    try_set(ff, 'codec', 'H264')
    try_set(ff, 'constant_rate_factor', 'HIGH')
    try_set(ff, 'ffmpeg_preset', 'GOOD')
    try_set(ff, 'gopsize', 12)
    try_set(ff, 'audio_codec', 'NONE')
    return path


def finish(scene):
    scene.frame_set(F_START)
    print("=" * 64)
    print("Scene built.  Frames %d-%d @ %d fps  |  %dx%d  |  %.1f s" %
          (F_START, F_END, FPS, RES_X, RES_Y, (F_END - F_START + 1) / float(FPS)))
    print("MP4 output: %s" % scene.render.filepath)
    if RENDER_ON_RUN or bpy.app.background:
        bpy.ops.render.render(animation=True)
        print("Render finished.")
    else:
        print("Render it with  Render > Render Animation  (Ctrl+F12).")
    print("=" * 64)


# ============================================================================
#  EXTRA HELPERS  (animation, rigging maths, shader odds and ends)
# ============================================================================
def iter_fcurves(idblock):
    """All F-curves of an object, on every Blender version (3.x ... 5.x)."""
    ad = getattr(idblock, 'animation_data', None)
    act = ad.action if ad is not None else None
    if act is None:
        return
    legacy = getattr(act, 'fcurves', None)
    if legacy is not None:
        for fc in legacy:
            yield fc
        return
    for layer in getattr(act, 'layers', []):
        for strip in getattr(layer, 'strips', []):
            for cb in getattr(strip, 'channelbags', []):
                for fc in cb.fcurves:
                    yield fc


@safe
def set_interp(ob, mode='LINEAR'):
    for fc in iter_fcurves(ob):
        for kp in fc.keyframe_points:
            kp.interpolation = mode


@safe
def add_noise_to_curves(ob, strength=0.1, scale=25.0, phase=0.0, only=None):
    """Adds a Noise F-curve modifier (organic wobble) to the object's animation."""
    n = 0
    for fc in iter_fcurves(ob):
        if only is not None and fc.data_path not in only:
            continue
        m = fc.modifiers.new('NOISE')
        m.strength = strength
        m.scale = scale
        m.phase = phase + n * 13.7
        n += 1


def map_range(nt, fac, a0, a1, b0=0.0, b1=1.0, loc=(0, 0), clamp_out=True):
    n = node(nt, 'ShaderNodeMapRange', loc)
    try_set(n, 'clamp', clamp_out)
    n.inputs['From Min'].default_value = a0
    n.inputs['From Max'].default_value = a1
    n.inputs['To Min'].default_value = b0
    n.inputs['To Max'].default_value = b1
    link(nt, fac, n.inputs['Value'])
    return n


# ---- small 3x3 matrix toolkit (rows = matrix rows) ---------------------------------------
def mat_cols(x, y, z):
    return ((x[0], y[0], z[0]), (x[1], y[1], z[1]), (x[2], y[2], z[2]))


def mat_vec(m, v):
    return (m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2])


def mat_mul(a, b):
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3))


def mat_T(m):
    return tuple(tuple(m[j][i] for j in range(3)) for i in range(3))


def mat_euler(e):
    """Blender 'XYZ' euler -> matrix (R = Rz * Ry * Rx)."""
    ca, sa = math.cos(e[0]), math.sin(e[0])
    cb, sb = math.cos(e[1]), math.sin(e[1])
    cc, sc = math.cos(e[2]), math.sin(e[2])
    rx = ((1, 0, 0), (0, ca, -sa), (0, sa, ca))
    ry = ((cb, 0, sb), (0, 1, 0), (-sb, 0, cb))
    rz = ((cc, -sc, 0), (sc, cc, 0), (0, 0, 1))
    return mat_mul(rz, mat_mul(ry, rx))


def euler_from_mat(m):
    sy = clamp(-m[2][0], -1.0, 1.0)
    b = math.asin(sy)
    if abs(sy) < 0.99999:
        a = math.atan2(m[2][1], m[2][2])
        c = math.atan2(m[1][0], m[0][0])
    else:
        a = math.atan2(-m[1][2], m[1][1])
        c = 0.0
    return (a, b, c)


def frame_neg_z(d, hint=(0.0, 0.0, 1.0)):
    """Rotation matrix whose local -Z axis points along d (local Y as close to `hint` as possible)."""
    z = vnorm(vmul(d, -1.0))
    x = vcross(hint, z)
    if vlen(x) < 1e-6:
        x = vcross((0.0, 1.0, 0.0), z)
    x = vnorm(x)
    y = vcross(z, x)
    return mat_cols(x, y, z)


def frame_pos_y(d, hint=(0.0, 0.0, 1.0)):
    """Rotation matrix whose local +Y axis points along d (local Z as close to `hint` as possible)."""
    y = vnorm(d)
    x = vcross(y, hint)
    if vlen(x) < 1e-6:
        x = vcross(y, (1.0, 0.0, 0.0))
    x = vnorm(x)
    z = vcross(x, y)
    return mat_cols(x, y, z)


def ik2_planar(l1, l2, ty, tz, bend):
    """Two-bone IK in the Y-Z plane. Bones hang along -Z; positive angle swings the bone toward +Y.
    bend=+1 -> joint ends up in FRONT of the hip-foot line, -1 -> BEHIND it.
    Returns (angle_upper, angle_lower_relative_to_upper)."""
    d0 = math.hypot(ty, tz)
    d = clamp(d0, abs(l1 - l2) + 1e-4, l1 + l2 - 1e-4)
    k = d / d0 if d0 > 1e-9 else 1.0
    ty, tz = ty * k, tz * k
    alpha = math.atan2(ty, -tz)
    g = math.acos(clamp((l1 * l1 + d * d - l2 * l2) / (2.0 * l1 * d), -1.0, 1.0))
    a1 = alpha + bend * g
    jy, jz = l1 * math.sin(a1), -l1 * math.cos(a1)
    a_low = math.atan2(ty - jy, -(tz - jz))
    return a1, a_low - a1


def ik2_3d(l1, l2, origin, target, pole):
    """Two-bone IK in 3D. Returns (elbow_position, reached_target)."""
    v = vsub(target, origin)
    d0 = vlen(v)
    d = clamp(d0, abs(l1 - l2) + 1e-4, l1 + l2 - 1e-4)
    dirn = vnorm(v)
    tgt = vadd(origin, vmul(dirn, d))
    a = (l1 * l1 - l2 * l2 + d * d) / (2.0 * d)
    h = math.sqrt(max(l1 * l1 - a * a, 0.0))
    pole_v = vsub(pole, vmul(dirn, vdot(pole, dirn)))
    pole_v = vnorm(pole_v) if vlen(pole_v) > 1e-6 else (0.0, 0.0, -1.0)
    elbow = vadd(vadd(origin, vmul(dirn, a)), vmul(pole_v, h))
    return elbow, tgt


def set_euler(ob, e):
    ob.rotation_euler = (e[0], e[1], e[2])


def key_trs(ob, frame, loc=True, rot=True, scl=False):
    if loc:
        ob.keyframe_insert(data_path='location', frame=frame)
    if rot:
        ob.keyframe_insert(data_path='rotation_euler', frame=frame)
    if scl:
        ob.keyframe_insert(data_path='scale', frame=frame)


def frames():
    return range(F_START, F_END + 1)


def tsec(f):
    return (f - F_START) / float(FPS)


def run_steps(steps):
    """Run build steps one by one; a failing optional step is reported but never stops the script."""
    for name, fn in steps:
        try:
            fn()
            print("[ok] " + name)
        except Exception:
            print("[FAILED] " + name + "  (the scene continues without it)")
            traceback.print_exc()


# ============================================================================
#  SHARED FX HELPERS  (floating particles, light shafts, ground, glow)
# ============================================================================
def find_node(m, bl_idname):
    for n in m.node_tree.nodes:
        if n.bl_idname == bl_idname:
            return n
    return None


def flicker(sock, base, amp, step=7, seed=0):
    """Key a node-socket value with random flicker so lights/particles feel alive."""
    r = random.Random(seed)
    f = F_START
    while True:
        sock.default_value = base * (1.0 + amp * (r.random() * 2.0 - 1.0))
        sock.keyframe_insert(data_path='default_value', frame=min(f, F_END))
        if f >= F_END:
            break
        f += step


def ellipsoid_geo(center, radii, rot=(0.0, 0.0, 0.0), segs=14, rings=9):
    v, f, uv = sphere_geo(segs, rings)
    out = []
    for p in v:
        q = rot_xyz((p[0] * radii[0], p[1] * radii[1], p[2] * radii[2]), rot)
        out.append((center[0] + q[0], center[1] + q[1], center[2] + q[2]))
    return out, f, uv


def make_motes(name, count, bmin, bmax, rmin, rmax, color, strength, seed, drift,
               wobble=0.15, flick=0.4):
    """Layer of tiny glowing particles (one merged mesh) that drifts and swirls gently."""
    rnd = random.Random(seed)
    mat = emission_mat(name + "Mat", color, strength)
    em = find_node(mat, 'ShaderNodeEmission')
    sv, sf, _ = sphere_geo(6, 4)
    mb = MeshBuilder()
    for _ in range(count):
        c = (rnd.uniform(bmin[0], bmax[0]), rnd.uniform(bmin[1], bmax[1]), rnd.uniform(bmin[2], bmax[2]))
        r = rnd.uniform(rmin, rmax)
        mb.add([vadd(c, vmul(p, r)) for p in sv], sf)
    ob = mb.build(name, mat, smooth=False, recalc=False)
    hide_from_rays(ob)
    for f in list(range(F_START, F_END + 1, 9)) + [F_END]:
        t = tsec(f)
        ob.location = (drift[0] * t + wobble * math.sin(1.3 * t + seed),
                       drift[1] * t + wobble * math.cos(0.9 * t + seed * 0.7),
                       drift[2] * t + wobble * 0.6 * math.sin(1.7 * t + seed * 0.3))
        ob.keyframe_insert(data_path='location', frame=f)
    if em is not None:
        flicker(em.inputs['Strength'], strength, flick, 6, seed)
    return ob


def shaft_material(name, color, strength, seed=0):
    m, nt = new_material(name, color)
    out = node(nt, 'ShaderNodeOutputMaterial', (1300, 0))
    tc = node(nt, 'ShaderNodeTexCoord', (-1100, 0))
    sep = node(nt, 'ShaderNodeSeparateXYZ', (-900, 0))
    link(nt, tc.outputs['UV'], sep.inputs['Vector'])
    u, v = sep.outputs['X'], sep.outputs['Y']
    a = math_node(nt, 'MULTIPLY', (-700, 150), a=u, b=2.0)
    b = math_node(nt, 'SUBTRACT', (-500, 150), a=a.outputs[0], b=1.0)
    c = math_node(nt, 'ABSOLUTE', (-300, 150), a=b.outputs[0])
    d = math_node(nt, 'SUBTRACT', (-100, 150), a=1.0, b=c.outputs[0], clamp_out=True)
    e = math_node(nt, 'POWER', (100, 150), a=d.outputs[0], b=1.7)
    f = math_node(nt, 'MULTIPLY', (-700, -50), a=v, b=math.pi)
    g = math_node(nt, 'SINE', (-500, -50), a=f.outputs[0])
    h = math_node(nt, 'POWER', (-300, -50), a=g.outputs[0], b=0.9)
    mask = math_node(nt, 'MULTIPLY', (300, 50), a=e.outputs[0], b=h.outputs[0])
    nz = node(nt, 'ShaderNodeTexNoise', (-700, -300))
    nz.inputs['Scale'].default_value = 5.0
    try_set(nz.inputs['Detail'], 'default_value', 3.0)
    mp = node(nt, 'ShaderNodeMapping', (-900, -300))
    mp.inputs['Location'].default_value = (seed * 3.1, seed * 1.7, 0.0)
    link(nt, tc.outputs['UV'], mp.inputs['Vector'])
    link(nt, mp.outputs['Vector'], nz.inputs['Vector'])
    streak = map_range(nt, nz.outputs['Fac'], 0.30, 0.70, 0.35, 1.0, (-400, -300))
    mask2 = math_node(nt, 'MULTIPLY', (500, 0), a=mask.outputs[0], b=streak.outputs['Result'])
    em = node(nt, 'ShaderNodeEmission', (700, -150))
    set_in(em, 'Color', rgba(color))
    set_in(em, 'Strength', strength)
    tr = node(nt, 'ShaderNodeBsdfTransparent', (700, 100))
    mx = node(nt, 'ShaderNodeMixShader', (1000, 0))
    link(nt, mask2.outputs[0], mx.inputs[0])
    link(nt, tr.outputs['BSDF'], mx.inputs[1])
    link(nt, em.outputs['Emission'], mx.inputs[2])
    link(nt, mx.outputs['Shader'], out.inputs['Surface'])
    set_blend(m)
    return m, em


def make_shafts(sun_dir, cam, count, xr, yr, length, wr, color, strength, seed, avoid=None, z0=0.0):
    """Soft god-ray beams. Each is a long strip lined up with the light direction that
    turns around its own axis to face the camera (Locked-Track constraint)."""
    rnd = random.Random(seed)
    d = vnorm(sun_dir)
    pitch = math.asin(clamp(d[2], -1.0, 1.0))
    yaw = math.atan2(-d[0], d[1])
    made = []
    for i in range(count):
        for _ in range(30):
            gx, gy = rnd.uniform(*xr), rnd.uniform(*yr)
            if avoid is None or not avoid(gx, gy):
                break
        w = rnd.uniform(*wr)
        L = length * rnd.uniform(0.8, 1.15)
        top = (gx - d[0] * L, gy - d[1] * L, z0 - d[2] * L)
        v = [(-w / 2, 0, 0), (w / 2, 0, 0), (w / 2, L, 0), (-w / 2, L, 0)]
        uv = [(0, 0), (1, 0), (1, 1), (0, 1)]
        mat, em = shaft_material("ShaftMat%d" % i, color, strength * rnd.uniform(0.6, 1.0), seed + i)
        ob = mesh_object("Shaft%d" % i, v, [(0, 1, 2, 3)], uv, mat, False, loc=top, rot=(pitch, 0.0, yaw),
                         recalc=False)
        c = ob.constraints.new('LOCKED_TRACK')
        c.target = cam
        c.track_axis = 'TRACK_Z'
        c.lock_axis = 'LOCK_Y'
        hide_from_rays(ob)
        flicker(em.inputs['Strength'], strength * rnd.uniform(0.6, 1.0), 0.35, 11, seed + i)
        made.append(ob)
    return made


def build_ground(name, cx, cy, half_x, half_y, nx, ny, hfun, mat):
    verts, uvs = [], []
    for j in range(ny + 1):
        for i in range(nx + 1):
            x = cx - half_x + 2.0 * half_x * i / nx
            y = cy - half_y + 2.0 * half_y * j / ny
            verts.append((x, y, hfun(x, y)))
            uvs.append((i / float(nx), j / float(ny)))
    faces = []
    for j in range(ny):
        for i in range(nx):
            a = j * (nx + 1) + i
            faces.append((a, a + 1, a + nx + 2, a + nx + 1))
    return mesh_object(name, verts, faces, uvs, mat, True, recalc=False)


def glow_sprite(name, size, color, strength, target, parent=None, loc=(0, 0, 0), power=2.0):
    m = glow_card_material(name + "Mat", color, strength, power)
    return billboard(name, size, m, target, parent, loc), m


# ============================================================================
#  PROCEDURAL NATURE  (grass, ferns, bushes, rocks)
# ============================================================================
def leafy_blob_geo(center, radius, rnd, segs=14, rings=8, amp=0.38, squash=1.0, freq=1.6):
    """Noise-displaced sphere - foliage clump / bush / boulder."""
    v, f, uv = sphere_geo(segs, rings)
    seed = rnd.randint(0, 9999)
    out = []
    for p in v:
        n = fbm(p[0] * freq + seed, p[1] * freq, p[2] * freq, 3, seed)
        k = 1.0 - amp * 0.5 + amp * n
        out.append((center[0] + p[0] * radius * k, center[1] + p[1] * radius * k,
                    center[2] + p[2] * radius * k * squash))
    return out, f, uv


def add_grass(mb, rnd, count, xr, yr, hfun, avoid, hr=(0.25, 0.6), wr=(0.012, 0.022)):
    n = 0
    tries = 0
    while n < count and tries < count * 4:
        tries += 1
        x, y = rnd.uniform(*xr), rnd.uniform(*yr)
        if avoid is not None and avoid(x, y):
            continue
        z = hfun(x, y) - 0.02
        h = rnd.uniform(*hr)
        w = rnd.uniform(*wr)
        a = rnd.uniform(0.0, 2.0 * math.pi)
        bend = rnd.uniform(0.12, 0.55) * h
        dx, dy = math.cos(a), math.sin(a)
        pts = [(x, y, z), (x + dx * bend * 0.2, y + dy * bend * 0.2, z + h * 0.5),
               (x + dx * bend, y + dy * bend, z + h)]
        hint = (-dy, dx, 0.25) if rnd.random() < 0.5 else (0.0, 0.0, 1.0)
        v, f, uv = ribbon_geo(pts, [w, w * 0.75, w * 0.06], hint)
        mb.add(v, f, uv)
        n += 1


def add_fern(mb, rnd, cx, cy, z, size, fronds=9, serration=0.55):
    for i in range(fronds):
        ang = 2.0 * math.pi * i / fronds + rnd.uniform(-0.3, 0.3)
        L = size * rnd.uniform(0.7, 1.2)
        dx, dy = math.cos(ang), math.sin(ang)
        stations = 13
        pts, widths = [], []
        for k in range(stations):
            u = k / float(stations - 1)
            r = L * u * 0.95
            h = L * (0.9 * u - 0.68 * u * u) * 1.1
            pts.append((cx + dx * r, cy + dy * r, z + h + 0.02))
            w = size * 0.26 * math.sin(math.pi * (u ** 0.8)) + 0.004
            widths.append(w * (1.0 if k % 2 == 0 else serration))
        v, f, uv = ribbon_geo(pts, widths, (0.0, 0.0, 1.0))
        mb.add(v, f, uv)


def rock_geo(center, size, rnd, squash=0.65):
    v, f, uv = sphere_geo(14, 9)
    seed = rnd.randint(0, 9999)
    out = []
    for p in v:
        n = fbm(p[0] * 1.9 + seed, p[1] * 1.9, p[2] * 1.9, 3, seed)
        k = 0.72 + 0.55 * n
        out.append((center[0] + p[0] * k * size * 1.1, center[1] + p[1] * k * size,
                    center[2] + max(p[2] * k, -0.45) * size * squash))
    return out, f, uv


def add_tuft_flowers(mb, rnd, count, xr, yr, hfun, avoid, r=0.028):
    sv, sf, _ = sphere_geo(6, 4)
    n = 0
    while n < count:
        x, y = rnd.uniform(*xr), rnd.uniform(*yr)
        if avoid is not None and avoid(x, y):
            continue
        z = hfun(x, y) + rnd.uniform(0.18, 0.4)
        rr = r * rnd.uniform(0.7, 1.3)
        mb.add([vadd((x, y, z), vmul(p, rr)) for p in sv], sf)
        n += 1


def plant_material(name, base, tip, rough=0.55, emit=0.08, uv_axis='Y'):
    """Green gradient along the blade/leaf length (UV)."""
    m, nt = new_material(name, base)
    out = node(nt, 'ShaderNodeOutputMaterial', (900, 0))
    b = node(nt, 'ShaderNodeBsdfPrincipled', (600, 0))
    link(nt, b.outputs['BSDF'], out.inputs['Surface'])
    tc = node(nt, 'ShaderNodeTexCoord', (-700, 0))
    sp = node(nt, 'ShaderNodeSeparateXYZ', (-500, 0))
    link(nt, tc.outputs['UV'], sp.inputs['Vector'])
    nz = node(nt, 'ShaderNodeTexNoise', (-500, -250))
    nz.inputs['Scale'].default_value = 3.0
    link(nt, tc.outputs['Object'], nz.inputs['Vector'])
    ramp = color_ramp(nt, [(0.0, base), (1.0, tip)], (-200, 0), fac=sp.outputs['X'])
    _, f, a, bb, o = mix_rgb(nt, 'MULTIPLY', (100, 0))
    f.default_value = 0.35
    link(nt, ramp.outputs['Color'], a)
    nr = color_ramp(nt, [(0.0, (0.55, 0.55, 0.55)), (1.0, (1.0, 1.0, 1.0))], (-200, -250), fac=nz.outputs['Fac'])
    link(nt, nr.outputs['Color'], bb)
    link(nt, o, b.inputs['Base Color'])
    set_in(b, 'Roughness', rough)
    set_in(b, ('Emission Color', 'Emission'), rgba(tip))
    set_in(b, 'Emission Strength', emit)
    return m


# ============================================================================
#  SCENE 1 :  KING OF THE FOREST
# ============================================================================
OUTPUT_NAME = "King_of_the_Forest.mp4"
USE_VOLUMETRICS = True                    # god-ray haze (slower). False = faster test renders
SUN_DIR = vnorm((0.40, -0.74, -0.48))     # direction the sunlight travels (sun sits behind the lion)
G = {}                                    # handles shared between build steps


def path_x(y):
    """World X of the forest path centre-line at world Y (the ground shader uses the same formula)."""
    return 0.20 * y - 0.10 + 0.50 * math.sin(0.95 * y + 0.4)


def hfun(x, y):
    """Ground height. Perfectly flat on the lion's path, gently hilly further away."""
    d = abs(x - path_x(y))
    k = smoothstep(1.3, 4.5, d)
    n = fbm(x * 0.30 + 10.0, y * 0.30 + 3.0, 0.0, 4, 3)
    n2 = fbm(x * 1.7, y * 1.7, 5.0, 2, 8)
    return k * (n * 1.1 + n2 * 0.14 - 0.45)


def near_path(w):
    return lambda x, y: abs(x - path_x(y)) < w


# ----------------------------------------------------------------------------
#  MATERIALS
# ----------------------------------------------------------------------------
def noise_material(name, stops, scale=2.0, detail=6.0, rough=0.7, bump=0.4, bump_scale=30.0,
                   mapping=None, top=None, top_range=(0.1, 0.85), emit=0.0, spec=0.3, view=None):
    """Generic procedural surface: noise -> colour ramp (+ colour on upward faces) + fine bump."""
    m, nt = new_material(name, view if view is not None else stops[len(stops) // 2][1])
    out = node(nt, 'ShaderNodeOutputMaterial', (1600, 0))
    b = node(nt, 'ShaderNodeBsdfPrincipled', (1300, 0))
    link(nt, b.outputs['BSDF'], out.inputs['Surface'])
    tc = node(nt, 'ShaderNodeTexCoord', (-1400, 0))
    vec = tc.outputs['Object']
    if mapping is not None:
        mp = node(nt, 'ShaderNodeMapping', (-1200, 0))
        mp.inputs['Scale'].default_value = mapping
        link(nt, tc.outputs['Object'], mp.inputs['Vector'])
        vec = mp.outputs['Vector']
    nz = node(nt, 'ShaderNodeTexNoise', (-1000, 200))
    nz.inputs['Scale'].default_value = scale
    try_set(nz.inputs['Detail'], 'default_value', detail)
    link(nt, vec, nz.inputs['Vector'])
    ramp = color_ramp(nt, stops, (-700, 200), fac=nz.outputs['Fac'])
    col = ramp.outputs['Color']
    if top is not None:
        ng = node(nt, 'ShaderNodeNewGeometry', (-1000, -200))
        sp = node(nt, 'ShaderNodeSeparateXYZ', (-800, -200))
        link(nt, ng.outputs['Normal'], sp.inputs['Vector'])
        mk = map_range(nt, sp.outputs['Z'], top_range[0], top_range[1], 0.0, 1.0, (-600, -200))
        _, f, a, bb, o = mix_rgb(nt, 'MIX', (-300, 0))
        link(nt, mk.outputs['Result'], f)
        link(nt, col, a)
        bb.default_value = rgba(top)
        col = o
    link(nt, col, b.inputs['Base Color'])
    set_in(b, 'Roughness', rough)
    if spec is not None:
        set_in(b, ('Specular IOR Level', 'Specular'), spec)
    if emit > 0:
        set_in(b, ('Emission Color', 'Emission'), rgba(stops[-1][1]))
        set_in(b, 'Emission Strength', emit)
    if bump > 0:
        n2 = node(nt, 'ShaderNodeTexNoise', (-1000, -500))
        n2.inputs['Scale'].default_value = bump_scale
        try_set(n2.inputs['Detail'], 'default_value', 8.0)
        link(nt, vec, n2.inputs['Vector'])
        bp = node(nt, 'ShaderNodeBump', (1000, -400))
        bp.inputs['Strength'].default_value = bump
        bp.inputs['Distance'].default_value = 0.02
        link(nt, n2.outputs['Fac'], bp.inputs['Height'])
        link(nt, bp.outputs['Normal'], b.inputs['Normal'])
    return m


def mat_ground():
    m, nt = new_material("ForestGround", (0.16, 0.20, 0.07))
    out = node(nt, 'ShaderNodeOutputMaterial', (1900, 0))
    b = node(nt, 'ShaderNodeBsdfPrincipled', (1600, 0))
    link(nt, b.outputs['BSDF'], out.inputs['Surface'])
    geo = node(nt, 'ShaderNodeNewGeometry', (-1800, 0))
    sep = node(nt, 'ShaderNodeSeparateXYZ', (-1600, 0))
    link(nt, geo.outputs['Position'], sep.inputs['Vector'])
    x, y = sep.outputs['X'], sep.outputs['Y']

    def noise(scale, detail, loc):
        n = node(nt, 'ShaderNodeTexNoise', loc)
        n.inputs['Scale'].default_value = scale
        try_set(n.inputs['Detail'], 'default_value', detail)
        link(nt, geo.outputs['Position'], n.inputs['Vector'])
        return n
    n1 = noise(0.8, 5.0, (-1400, 500))
    n2 = noise(6.0, 4.0, (-1400, 250))
    n3 = noise(45.0, 8.0, (-1400, -450))
    # path mask:  distance from  x = 0.2*y - 0.1 + 0.5*sin(0.95*y + 0.4)
    t1 = math_node(nt, 'MULTIPLY', (-1400, -100), a=y, b=0.95)
    t2 = math_node(nt, 'ADD', (-1200, -100), a=t1.outputs[0], b=0.4)
    t3 = math_node(nt, 'SINE', (-1000, -100), a=t2.outputs[0])
    t4 = math_node(nt, 'MULTIPLY', (-800, -100), a=t3.outputs[0], b=0.5)
    l1 = math_node(nt, 'MULTIPLY', (-1400, -300), a=y, b=0.2)
    l2 = math_node(nt, 'SUBTRACT', (-1200, -300), a=l1.outputs[0], b=0.1)
    px = math_node(nt, 'ADD', (-600, -200), a=l2.outputs[0], b=t4.outputs[0])
    dx = math_node(nt, 'SUBTRACT', (-400, -200), a=x, b=px.outputs[0])
    ad = math_node(nt, 'ABSOLUTE', (-200, -200), a=dx.outputs[0])
    j1 = math_node(nt, 'SUBTRACT', (-1000, 250), a=n2.outputs['Fac'], b=0.5)
    j2 = math_node(nt, 'MULTIPLY', (-800, 250), a=j1.outputs[0], b=0.9)
    dist = math_node(nt, 'ADD', (0, -200), a=ad.outputs[0], b=j2.outputs[0])
    mask = map_range(nt, dist.outputs[0], 0.55, 1.2, 1.0, 0.0, (200, -200))
    grass = color_ramp(nt, [(0.0, (0.04, 0.09, 0.025)), (0.5, (0.10, 0.18, 0.04)),
                            (0.75, (0.22, 0.20, 0.07)), (1.0, (0.28, 0.17, 0.06))], (-1000, 500), fac=n1.outputs['Fac'])
    dirt = color_ramp(nt, [(0.0, (0.24, 0.17, 0.09)), (1.0, (0.42, 0.31, 0.18))], (-1000, -450), fac=n3.outputs['Fac'])
    _, f, a, bb, o = mix_rgb(nt, 'MIX', (400, 100))
    link(nt, mask.outputs['Result'], f)
    link(nt, grass.outputs['Color'], a)
    link(nt, dirt.outputs['Color'], bb)
    link(nt, o, b.inputs['Base Color'])
    set_in(b, 'Roughness', 0.95)
    set_in(b, ('Specular IOR Level', 'Specular'), 0.15)
    bp = node(nt, 'ShaderNodeBump', (1300, -300))
    bp.inputs['Strength'].default_value = 0.6
    bp.inputs['Distance'].default_value = 0.03
    link(nt, n3.outputs['Fac'], bp.inputs['Height'])
    link(nt, bp.outputs['Normal'], b.inputs['Normal'])
    return m


def mat_bark():
    return noise_material("Bark", [(0.25, (0.045, 0.032, 0.024)), (0.55, (0.15, 0.11, 0.075)), (0.8, (0.27, 0.21, 0.15))],
                          scale=1.0, detail=7.0, rough=0.9, bump=0.9, bump_scale=9.0,
                          mapping=(7.0, 7.0, 0.9), view=(0.15, 0.10, 0.07), spec=0.1)


def mat_leaves():
    return noise_material("Leaves", [(0.0, (0.025, 0.07, 0.02)), (0.5, (0.09, 0.22, 0.04)), (1.0, (0.30, 0.42, 0.07))],
                          scale=1.3, detail=4.0, rough=0.5, bump=0.5, bump_scale=14.0,
                          top=(0.38, 0.5, 0.10), top_range=(0.0, 0.95), emit=0.06, view=(0.10, 0.25, 0.05))


def mat_rock():
    return noise_material("Rock", [(0.3, (0.10, 0.10, 0.095)), (0.6, (0.22, 0.21, 0.19)), (0.85, (0.34, 0.32, 0.28))],
                          scale=3.0, detail=8.0, rough=0.85, bump=1.0, bump_scale=18.0,
                          top=(0.07, 0.17, 0.04), top_range=(0.35, 0.9), view=(0.2, 0.2, 0.18), spec=0.2)


# ----------------------------------------------------------------------------
#  TREES
# ----------------------------------------------------------------------------
def make_tree(bark, leaves, rnd, x, y, height, radius, foliage=True):
    z0 = hfun(x, y)
    n = 14
    pts, rad = [], []
    phase = rnd.uniform(0, 6.28)
    lean = (rnd.uniform(-1, 1) * 0.05 * height, rnd.uniform(-1, 1) * 0.05 * height)
    for i in range(n):
        u = i / float(n - 1)
        pts.append((x + lean[0] * u * u + 0.18 * math.sin(u * 4.0 + phase) * u,
                    y + lean[1] * u * u + 0.18 * math.cos(u * 3.3 + phase) * u,
                    z0 - 0.4 + (height + 0.4) * u))
        rad.append(radius * (1.0 + 1.25 * math.exp(-u * 16.0)) * (1.0 - 0.5 * u))
    v, f, uv = tube_geo(pts, rad, 10, cap_start=False, cap_end=True)
    bark.add(v, f, uv)
    for k in range(5):                                           # root flares
        a = rnd.uniform(0, 6.28)
        rl = radius * rnd.uniform(2.2, 3.4)
        p0 = (x + math.cos(a) * radius * 0.5, y + math.sin(a) * radius * 0.5, z0 + 0.55)
        p1 = (x + math.cos(a) * rl * 0.55, y + math.sin(a) * rl * 0.55, z0 + 0.12)
        p2 = (x + math.cos(a) * rl, y + math.sin(a) * rl, z0 - 0.12)
        v, f, uv = tube_geo(spline([p0, p1, p2], 6), [radius * 0.32 * (1 - 0.85 * i / 5.0) for i in range(6)], 6)
        bark.add(v, f, uv)
    for k in range(rnd.randint(4, 7)):                           # branches + foliage clumps
        u0 = rnd.uniform(0.5, 0.9)
        base = pts[int(u0 * (n - 1))]
        a = rnd.uniform(0, 6.28)
        L = rnd.uniform(2.5, 5.0) * (0.6 + (1.0 - u0))
        end = (base[0] + math.cos(a) * L, base[1] + math.sin(a) * L, base[2] + rnd.uniform(0.3, 2.2))
        mid = ((base[0] + end[0]) / 2, (base[1] + end[1]) / 2, base[2] + (end[2] - base[2]) * 0.35 + 0.5)
        v, f, uv = tube_geo(spline([base, mid, end], 7), [radius * 0.22 * (1 - 0.9 * i / 6.0) + 0.015 for i in range(7)], 6)
        bark.add(v, f, uv)
        if foliage:
            for j in range(rnd.randint(2, 3)):
                c = (end[0] + rnd.uniform(-1, 1), end[1] + rnd.uniform(-1, 1), end[2] + rnd.uniform(-0.3, 1.2))
                leaves.add(*leafy_blob_geo(c, rnd.uniform(1.1, 2.2), rnd, 12, 7, 0.55, 0.75))
    if foliage:
        top = pts[-1]
        for j in range(rnd.randint(3, 5)):
            c = (top[0] + rnd.uniform(-2, 2), top[1] + rnd.uniform(-2, 2), top[2] - rnd.uniform(0, 3))
            leaves.add(*leafy_blob_geo(c, rnd.uniform(2.2, 3.8), rnd, 12, 7, 0.6, 0.7))


def step_trees():
    rnd = random.Random(21)
    bark, leaves = MeshBuilder(), MeshBuilder()
    placed = []

    def ok(x, y, r, corridor):
        if abs(x - path_x(y)) < corridor:
            return False
        for (px, py, pr) in placed:
            if math.hypot(x - px, y - py) < (r + pr) * 2.2 + 0.6:
                return False
        return True
    n = tries = 0
    while n < 30 and tries < 900:                                # close / mid trees
        tries += 1
        y = rnd.uniform(-3.0, 30.0)
        x = rnd.choice((-1, 1)) * rnd.uniform(2.8, 13.0) + path_x(y) * 0.5
        r = rnd.uniform(0.28, 0.55)
        if ok(x, y, r, 2.8):
            placed.append((x, y, r))
            make_tree(bark, leaves, rnd, x, y, rnd.uniform(11.0, 16.0), r, True)
            n += 1
    n = tries = 0
    while n < 30 and tries < 900:                                # distant forest wall (fades into haze)
        tries += 1
        y = rnd.uniform(30.0, 75.0)
        x = rnd.uniform(-30.0, 30.0)
        r = rnd.uniform(0.4, 0.7)
        if ok(x, y, r, 3.0):
            placed.append((x, y, r))
            make_tree(bark, leaves, rnd, x, y, rnd.uniform(14.0, 20.0), r, True)
            n += 1
    G['bark'] = bark.build("TreeBark", mat_bark(), smooth=True)
    G['leaves'] = leaves.build("TreeLeaves", mat_leaves(), smooth=True)
    try_set(G['leaves'], 'visible_glossy', False)


# ----------------------------------------------------------------------------
#  GROUND + UNDERGROWTH
# ----------------------------------------------------------------------------
def step_ground():
    G['ground'] = build_ground("Ground", 0.0, 25.0, 45.0, 40.0, 100, 88, hfun, mat_ground())


def step_undergrowth():
    rnd = random.Random(33)
    away = near_path(0.85)
    grass_mat = plant_material("Grass", (0.04, 0.12, 0.02), (0.45, 0.55, 0.12), 0.5, 0.12)
    fern_mat = plant_material("Fern", (0.03, 0.14, 0.03), (0.30, 0.52, 0.10), 0.5, 0.10)
    g1 = MeshBuilder()
    add_grass(g1, rnd, 5200, (-9, 9), (-2.5, 14), hfun, away)
    add_grass(g1, rnd, 2600, (-14, 14), (14, 40), hfun, away, hr=(0.35, 0.9), wr=(0.02, 0.035))
    G['grass'] = g1.build("Grass", grass_mat, smooth=True)
    fe = MeshBuilder()
    n = 0
    while n < 42:
        y = rnd.uniform(-2.0, 24.0)
        x = rnd.uniform(-10.0, 10.0)
        if abs(x - path_x(y)) < 1.3:
            continue
        add_fern(fe, rnd, x, y, hfun(x, y), rnd.uniform(0.5, 1.0), rnd.randint(7, 11))
        n += 1
    G['ferns'] = fe.build("Ferns", fern_mat, smooth=True)
    bu = MeshBuilder()
    n = 0
    while n < 12:
        y = rnd.uniform(1.0, 26.0)
        x = rnd.choice((-1, 1)) * rnd.uniform(3.6, 10.0) + path_x(y) * 0.5
        base = hfun(x, y)
        for j in range(4):
            bu.add(*leafy_blob_geo((x + rnd.uniform(-0.6, 0.6), y + rnd.uniform(-0.6, 0.6), base + rnd.uniform(0.3, 0.8)),
                                   rnd.uniform(0.5, 1.0), rnd, 12, 7, 0.5, 0.8))
        n += 1
    G['bushes'] = bu.build("Bushes", mat_leaves(), smooth=True)
    ro = MeshBuilder()
    for i in range(16):
        y = rnd.uniform(-1.5, 22.0)
        x = rnd.uniform(-9.0, 9.0)
        if abs(x - path_x(y)) < 1.4:
            continue
        s = rnd.uniform(0.2, 0.8) if i < 14 else rnd.uniform(1.3, 1.8)
        ro.add(*rock_geo((x, y, hfun(x, y) + s * 0.1), s, rnd))
    G['rocks'] = ro.build("Rocks", mat_rock(), smooth=True)
    # fallen log
    p0, p1 = (-4.6, 6.2), (-2.9, 9.0)
    pts = [(p0[0] + (p1[0] - p0[0]) * i / 7.0, p0[1] + (p1[1] - p0[1]) * i / 7.0,
            hfun(p0[0] + (p1[0] - p0[0]) * i / 7.0, p0[1] + (p1[1] - p0[1]) * i / 7.0) + 0.26) for i in range(8)]
    v, f, uv = tube_geo(pts, [0.27, 0.29, 0.30, 0.29, 0.30, 0.28, 0.27, 0.24], 12)
    G['log'] = mesh_object("FallenLog", v, f, uv, G['bark'].data.materials[0] if G.get('bark') else mat_bark(), True)
    fl = MeshBuilder()
    add_tuft_flowers(fl, rnd, 110, (-8, 8), (-1.5, 16), hfun, near_path(0.9))
    G['flowers'] = fl.build("Flowers", emission_mat("FlowerMat", (1.0, 0.85, 0.35), 0.8), smooth=False, recalc=False)


# ----------------------------------------------------------------------------
#  WORLD, SUN, HAZE
# ----------------------------------------------------------------------------
def step_world():
    make_sky((0.28, 0.45, 0.60), (1.0, 0.80, 0.52), (0.06, 0.05, 0.03), sun_dir=SUN_DIR,
             sun_color=(1.0, 0.78, 0.50), glow=1.6, glow_power=5.0, disc=2.5, stars=0.0, strength=1.0, band=0.35)
    G['sun'] = add_light('SUN', 'Sun', (0, 20, 30), 5.0, (1.0, 0.80, 0.52), size=0.03, direction=SUN_DIR)
    if USE_VOLUMETRICS:
        G['haze'] = add_volume_box((0, 22, 7), (70, 90, 15), 0.016, (1.0, 0.95, 0.85), aniso=0.55)


# ----------------------------------------------------------------------------
#  LION MATERIALS
# ----------------------------------------------------------------------------
def mat_fur():
    m, nt = new_material("LionFur", (0.55, 0.36, 0.17))
    out = node(nt, 'ShaderNodeOutputMaterial', (1800, 0))
    b = node(nt, 'ShaderNodeBsdfPrincipled', (1500, 0))
    link(nt, b.outputs['BSDF'], out.inputs['Surface'])
    tc = node(nt, 'ShaderNodeTexCoord', (-1600, 0))
    ng = node(nt, 'ShaderNodeNewGeometry', (-1600, -400))
    sp = node(nt, 'ShaderNodeSeparateXYZ', (-1400, -400))
    link(nt, ng.outputs['Normal'], sp.inputs['Vector'])
    n1 = node(nt, 'ShaderNodeTexNoise', (-1400, 300))
    n1.inputs['Scale'].default_value = 8.0
    try_set(n1.inputs['Detail'], 'default_value', 4.0)
    link(nt, tc.outputs['Object'], n1.inputs['Vector'])
    n2 = node(nt, 'ShaderNodeTexNoise', (-1400, 50))
    n2.inputs['Scale'].default_value = 90.0
    try_set(n2.inputs['Detail'], 'default_value', 8.0)
    link(nt, tc.outputs['Object'], n2.inputs['Vector'])
    base = color_ramp(nt, [(0.25, (0.36, 0.20, 0.07)), (0.55, (0.56, 0.36, 0.15)), (0.85, (0.68, 0.47, 0.22))],
                      (-1100, 300), fac=n1.outputs['Fac'])
    belly = map_range(nt, sp.outputs['Z'], -0.65, 0.05, 1.0, 0.0, (-1100, -400))
    _, f1, a1, b1, o1 = mix_rgb(nt, 'MIX', (-800, 100))
    link(nt, belly.outputs['Result'], f1)
    link(nt, base.outputs['Color'], a1)
    b1.default_value = (0.74, 0.60, 0.38, 1.0)
    grain = color_ramp(nt, [(0.0, (0.62, 0.62, 0.62)), (1.0, (1.0, 1.0, 1.0))], (-1100, -50), fac=n2.outputs['Fac'])
    _, f2, a2, b2, o2 = mix_rgb(nt, 'MULTIPLY', (-500, 100))
    f2.default_value = 0.8
    link(nt, o1, a2)
    link(nt, grain.outputs['Color'], b2)
    link(nt, o2, b.inputs['Base Color'])
    set_in(b, 'Roughness', 0.62)
    set_in(b, ('Specular IOR Level', 'Specular'), 0.25)
    set_in(b, ('Sheen Weight', 'Sheen'), 0.7)
    set_in(b, 'Sheen Roughness', 0.35)
    bp = node(nt, 'ShaderNodeBump', (1200, -300))
    bp.inputs['Strength'].default_value = 0.45
    bp.inputs['Distance'].default_value = 0.003
    link(nt, n2.outputs['Fac'], bp.inputs['Height'])
    link(nt, bp.outputs['Normal'], b.inputs['Normal'])
    return m


def mat_mane():
    m, nt = new_material("LionMane", (0.38, 0.20, 0.07))
    out = node(nt, 'ShaderNodeOutputMaterial', (1200, 0))
    b = node(nt, 'ShaderNodeBsdfPrincipled', (900, 0))
    link(nt, b.outputs['BSDF'], out.inputs['Surface'])
    tc = node(nt, 'ShaderNodeTexCoord', (-900, 0))
    sp = node(nt, 'ShaderNodeSeparateXYZ', (-700, 0))
    link(nt, tc.outputs['UV'], sp.inputs['Vector'])
    oi = node(nt, 'ShaderNodeObjectInfo', (-700, -300))
    ramp = color_ramp(nt, [(0.0, (0.62, 0.40, 0.16)), (0.45, (0.30, 0.15, 0.055)), (1.0, (0.06, 0.03, 0.015))],
                      (-400, 100), fac=sp.outputs['X'])
    rnd_c = color_ramp(nt, [(0.0, (0.55, 0.50, 0.45)), (1.0, (1.0, 0.95, 0.85))], (-400, -300), fac=oi.outputs['Random'])
    _, f, a, bb, o = mix_rgb(nt, 'MULTIPLY', (-100, 0))
    f.default_value = 1.0
    link(nt, ramp.outputs['Color'], a)
    link(nt, rnd_c.outputs['Color'], bb)
    link(nt, o, b.inputs['Base Color'])
    set_in(b, 'Roughness', 0.45)
    set_in(b, ('Specular IOR Level', 'Specular'), 0.35)
    set_in(b, ('Sheen Weight', 'Sheen'), 0.5)
    return m


def mat_eye():
    m, nt = new_material("LionEye", (0.8, 0.5, 0.1))
    out = node(nt, 'ShaderNodeOutputMaterial', (1100, 0))
    b = node(nt, 'ShaderNodeBsdfPrincipled', (800, 0))
    link(nt, b.outputs['BSDF'], out.inputs['Surface'])
    tc = node(nt, 'ShaderNodeTexCoord', (-900, 0))
    sp = node(nt, 'ShaderNodeSeparateXYZ', (-700, 0))
    link(nt, tc.outputs['Object'], sp.inputs['Vector'])
    x2 = math_node(nt, 'MULTIPLY', (-500, 100), a=sp.outputs['X'], b=sp.outputs['X'])
    z2 = math_node(nt, 'MULTIPLY', (-500, -100), a=sp.outputs['Z'], b=sp.outputs['Z'])
    s = math_node(nt, 'ADD', (-300, 0), a=x2.outputs[0], b=z2.outputs[0])
    r = math_node(nt, 'SQRT', (-100, 0), a=s.outputs[0])
    ramp = color_ramp(nt, [(0.0, (0.01, 0.008, 0.005)), (0.27, (0.01, 0.008, 0.005)), (0.31, (0.95, 0.65, 0.12)),
                           (0.50, (0.75, 0.42, 0.06)), (0.60, (0.20, 0.10, 0.03)), (0.66, (0.06, 0.04, 0.03)),
                           (1.0, (0.06, 0.04, 0.03))], (200, 0), fac=r.outputs[0])
    link(nt, ramp.outputs['Color'], b.inputs['Base Color'])
    for nm in ('Emission Color', 'Emission'):
        if nm in b.inputs:
            link(nt, ramp.outputs['Color'], b.inputs[nm])
            break
    set_in(b, 'Emission Strength', 0.25)
    set_in(b, 'Roughness', 0.08)
    set_in(b, ('Specular IOR Level', 'Specular'), 0.9)
    set_in(b, ('Coat Weight', 'Clearcoat'), 1.0)
    return m


# ----------------------------------------------------------------------------
#  LION BODY  (empties = joints, merged meshes = body parts)
# ----------------------------------------------------------------------------
Z0 = 0.88                 # root (torso centre) height above ground
CHEST_Y = -0.04           # chest joint, in root frame
SH_Y, HIP_Y = 0.52, -0.52
BETA = 0.75               # fraction of the stride a paw spends on the ground
FRONT = dict(l1=0.37, l2=0.38, l3=0.12, bend=-1, psi0=0.0, lift=0.09, yc=0.03, hz=-0.06, k=0.5)
HIND = dict(l1=0.40, l2=0.38, l3=0.22, bend=1, psi0=0.28, lift=0.10, yc=-0.04, hz=-0.04, k=0.7)


def mane_cluster_geo(rnd, strands, length, outd, flow=(0.0, -0.55, -0.80), curl=1.0, spread=0.28):
    """A clump of tapered locks. They leave the skin along `outd`, then sweep toward `flow`
    (back + down). Local origin = clump root, so the whole clump can sway around it."""
    mb = MeshBuilder()
    for s in range(strands):
        o = (rnd.uniform(-0.03, 0.03), rnd.uniform(-0.03, 0.03), rnd.uniform(-0.03, 0.03))
        d_out = vnorm(vadd(outd, (rnd.uniform(-spread, spread), rnd.uniform(-spread, spread), rnd.uniform(-spread, spread))))
        fl = vnorm(vadd(flow, (rnd.uniform(-0.2, 0.2), rnd.uniform(-0.2, 0.2), rnd.uniform(-0.2, 0.2))))
        L = length * rnd.uniform(0.75, 1.1)
        n = 10
        ph = rnd.uniform(0.0, 6.28)
        pts = [o]
        for k in range(1, n):
            w = clamp(smoothstep(0.0, 1.0, (k - 1) / (n - 2.0)) * curl)
            d = vnorm(vlerp(d_out, fl, w))
            side = vcross(d, (0.0, 0.0, 1.0))
            d = vnorm(vadd(d, vmul(side, 0.063 * math.sin(k * 1.1 + ph))))
            pts.append(vadd(pts[-1], vmul(d, L / (n - 1.0))))
        rad = [0.0068 * (1.0 - k / (n - 1.0)) + 0.0009 for k in range(n)]
        v, f, uv = tube_geo(pts, rad, 4, False, True)
        mb.add(v, f, uv)
    return mb.v, mb.f, mb.uv


def build_leg(name, front, x, parent, hy, fur):
    sp = FRONT if front else HIND
    l1, l2, l3 = sp['l1'], sp['l2'], sp['l3']
    hip = empty(name + "_hip", (x, hy, sp['hz']), parent=parent)
    knee = empty(name + "_knee", (0, 0, -l1), parent=hip)
    ankle = empty(name + "_ankle", (0, 0, -l2), parent=knee)
    paw = empty(name + "_paw", (0, 0, -l3), parent=ankle)
    mb = MeshBuilder()
    if front:
        mb.add(*tube_geo([(0, 0, 0), (0, 0, -l1 * 0.5), (0, 0, -l1)], [0.125, 0.11, 0.092], 10))
        mb.add(*ellipsoid_geo((0, 0.0, -0.10), (0.135, 0.17, 0.20)))
        mb.add(*ellipsoid_geo((0, -0.07, -l1 + 0.02), (0.07, 0.08, 0.08)))
    else:
        mb.add(*tube_geo([(0, 0, 0), (0, 0, -l1 * 0.5), (0, 0, -l1)], [0.14, 0.12, 0.092], 10))
        mb.add(*ellipsoid_geo((0, -0.03, -0.10), (0.15, 0.22, 0.23)))
    mb.build(name + "_upper", fur, True, hip)
    mb = MeshBuilder()
    rr = [0.092, 0.075, 0.062] if front else [0.10, 0.078, 0.058]
    mb.add(*tube_geo([(0, 0, 0), (0, 0, -l2 * 0.5), (0, 0, -l2)], rr, 10))
    mb.build(name + "_lower", fur, True, knee)
    mb = MeshBuilder()
    mb.add(*tube_geo([(0, 0, 0), (0, 0, -l3)], [0.062, 0.068], 10))
    mb.build(name + "_foot", fur, True, ankle)
    mb = MeshBuilder()
    mb.add(*ellipsoid_geo((0, 0.045, -0.03), (0.072, 0.115, 0.05)))
    for tx in (-0.04, -0.013, 0.013, 0.04):
        mb.add(*ellipsoid_geo((tx, 0.125, -0.036), (0.026, 0.03, 0.028)))
    mb.build(name + "_pawmesh", fur, True, paw)
    return dict(hip=hip, knee=knee, ankle=ankle, paw=paw, sp=sp)


def step_lion():
    fur, mane_m, eye_m = mat_fur(), mat_mane(), mat_eye()
    dark = pbr("LionDark", (0.045, 0.028, 0.018), 0.75)
    nose_m = pbr("LionNose", (0.12, 0.05, 0.05), 0.22, spec=0.6)
    inner = pbr("EarInner", (0.35, 0.20, 0.15), 0.7)
    whisk = pbr("Whisker", (0.9, 0.86, 0.75), 0.4)
    L = {}
    rnd = random.Random(5)
    root = empty("LionRoot", (0, 0, Z0), size=0.4)
    L['root'] = root
    # ---- torso, rear half ----
    ys = [-0.94, -0.88, -0.78, -0.62, -0.42, -0.22, -0.04, 0.06]
    rz = [0.05, 0.17, 0.25, 0.29, 0.29, 0.27, 0.265, 0.265]
    cz = [0.0, 0.03, 0.05, 0.04, 0.02, 0.0, 0.0, 0.0]
    mb = MeshBuilder()
    mb.add(*tube_geo([(0.0, y, c) for y, c in zip(ys, cz)], rz, 14, True, True, 0.95))
    for s in (-1, 1):
        mb.add(*ellipsoid_geo((s * 0.20, -0.55, -0.03), (0.13, 0.30, 0.25)))
    mb.build("LionRear", fur, True, root)
    # ---- torso, front half (chest joint allows a little spine flex) ----
    chest = empty("LionChest", (0, CHEST_Y, 0), parent=root)
    L['chest'] = chest
    ys = [-0.10, 0.10, 0.35, 0.58, 0.78, 0.92, 1.0, 1.05]
    rz = [0.265, 0.285, 0.30, 0.32, 0.30, 0.22, 0.12, 0.04]
    cz = [0.0, 0.0, 0.0, -0.01, 0.0, 0.02, 0.05, 0.07]
    mb = MeshBuilder()
    mb.add(*tube_geo([(0.0, y, c) for y, c in zip(ys, cz)], rz, 14, True, True, 0.9))
    mb.add(*ellipsoid_geo((0, 0.42, 0.16), (0.19, 0.30, 0.10)))
    mb.add(*ellipsoid_geo((0, 0.85, 0.02), (0.24, 0.24, 0.29)))
    L['chest_mesh'] = mb.build("LionChestMesh", fur, True, chest)
    # ---- neck + head ----
    neck = empty("LionNeck", (0, 0.78, 0.10), parent=chest)
    head = empty("LionHead", (0, 0.32, 0.05), parent=neck)
    L['neck'], L['head'] = neck, head
    head.scale = (1.2, 1.2, 1.2)                                            # lions have big heads
    pts = spline([(0, -0.08, -0.04), (0, 0.12, 0.0), (0, 0.32, 0.05)], 6)
    v, f, uv = tube_geo(pts, [0.2, 0.2, 0.19, 0.18, 0.17, 0.17], 12, True, True, 1.0)
    mesh_object("LionNeckMesh", v, f, uv, fur, True, neck)
    mb = MeshBuilder()
    mb.add(*ellipsoid_geo((0, -0.02, 0.0), (0.155, 0.19, 0.16)))
    mb.add(*ellipsoid_geo((0, 0.17, -0.055), (0.088, 0.14, 0.075)))
    mb.add(*ellipsoid_geo((0, 0.20, 0.01), (0.05, 0.12, 0.05)))
    mb.add(*ellipsoid_geo((0, -0.10, -0.04), (0.21, 0.19, 0.22)))
    for s in (-1, 1):
        mb.add(*ellipsoid_geo((s * 0.105, 0.04, -0.075), (0.075, 0.095, 0.085)))
        mb.add(*ellipsoid_geo((s * 0.066, 0.10, 0.066), (0.055, 0.075, 0.032), (0, 0, -s * 0.25)))
        mb.add(*ellipsoid_geo((s * 0.05, 0.22, -0.06), (0.047, 0.05, 0.032)))
    mb.build("LionHeadMesh", fur, True, head)
    blob("LionNose", (0, 0.305, -0.015), (0.045, 0.032, 0.034), (0, 0, 0), nose_m, head, 12, 8)
    mb = MeshBuilder()                                                       # mouth lines
    mb.add(*tube_geo([(0, 0.305, -0.04), (0, 0.312, -0.085)], [0.0035, 0.0035], 5))
    for s in (-1, 1):
        pts = spline([(0, 0.312, -0.085), (s * 0.03, 0.30, -0.105), (s * 0.075, 0.24, -0.115), (s * 0.095, 0.17, -0.108)], 8)
        mb.add(*tube_geo(pts, [0.0038] * 8, 5))
    mb.build("LionMouth", dark, True, head)
    jaw = empty("LionJaw", (0, 0.06, -0.09), parent=head)
    L['jaw'] = jaw
    mb = MeshBuilder()
    mb.add(*ellipsoid_geo((0, 0.12, -0.02), (0.062, 0.12, 0.036)))
    mb.add(*ellipsoid_geo((0, 0.22, -0.035), (0.04, 0.04, 0.03)))
    mb.build("LionJawMesh", fur, True, jaw)
    L['lids'] = []
    for s, side in ((-1, 'L'), (1, 'R')):
        blob("LionEye_" + side, (s * 0.085, 0.133, 0.035), (0.021, 0.021, 0.021), (0, 0, 0), eye_m, head, 16, 12)
        L['lids'].append(blob("LionLid_" + side, (s * 0.085, 0.136, 0.078), (0.0235, 0.0235, 0.0235), (0, 0, 0),
                              fur, head, 12, 8))
    L['ears'] = []
    for s, side in ((-1, 'L'), (1, 'R')):
        ej = empty("LionEar_" + side, (s * 0.118, -0.07, 0.12), parent=head)
        mb = MeshBuilder()
        mb.add(*ellipsoid_geo((0, 0, 0.03), (0.048, 0.024, 0.052), (0, 0, s * 0.35)))
        mb.build("LionEarMesh_" + side, fur, True, ej)
        mesh_object("LionEarBack_" + side, *ellipsoid_geo((0, -0.012, 0.03), (0.046, 0.016, 0.05), (0, 0, s * 0.35)),
                    mat=dark, smooth=True, parent=ej)
        mesh_object("LionEarInner_" + side, *ellipsoid_geo((0, 0.012, 0.03), (0.03, 0.012, 0.034), (0, 0, s * 0.35)),
                    mat=inner, smooth=True, parent=ej)
        L['ears'].append((ej, s))
    mb = MeshBuilder()                                                       # whiskers
    for s in (-1, 1):
        for j in range(7):
            p0 = (s * 0.06, 0.24, -0.052 + 0.011 * (j % 4))
            dd = vnorm((s * 0.9, 0.55, -0.12 + 0.07 * j - 0.25))
            ln = rnd.uniform(0.12, 0.17)
            p1 = vadd(p0, vmul(dd, ln * 0.5))
            p2 = vadd(vadd(p0, vmul(dd, ln)), (0, 0, -0.025))
            mb.add(*tube_geo(spline([p0, p1, p2], 5), [0.0013, 0.001, 0.0006, 0.0005, 0.0003], 4))
    mb.build("LionWhiskers", whisk, True, head)
    # ---- mane ----
    L['mane'] = []
    n = 0
    for i in range(46):                                                      # around the head
        for _ in range(80):
            uy = rnd.uniform(-0.95, 0.25)
            a = rnd.uniform(0, 2 * math.pi)
            rr = math.sqrt(1 - uy * uy)
            u = (rr * math.cos(a), uy, rr * math.sin(a))
            if uy > 0.0 and abs(u[0]) < 0.45 and u[2] > -0.5:
                continue
            break
        rootp = (u[0] * 0.19, -0.06 + u[1] * 0.16, -0.01 + u[2] * 0.20)
        outd = vnorm((u[0], u[1] * 0.6 - 0.35, u[2]))
        v, f, uv = mane_cluster_geo(rnd, 14, rnd.uniform(0.30, 0.50), outd)
        L['mane'].append((mesh_object("ManeH%d" % i, v, f, uv, mane_m, True, head, loc=rootp), n))
        n += 1
    for i in range(16):                                                      # cheek ruff framing the face
        s = rnd.choice((-1, 1))
        rootp = (s * rnd.uniform(0.10, 0.15), rnd.uniform(0.0, 0.12), rnd.uniform(-0.12, 0.12))
        v, f, uv = mane_cluster_geo(rnd, 9, rnd.uniform(0.12, 0.2), vnorm((s * 1.0, -0.2, rnd.uniform(-0.1, 0.3))), (0.0, -0.7, -0.3), 0.9)
        L['mane'].append((mesh_object("ManeC%d" % i, v, f, uv, mane_m, True, head, loc=rootp), n))
        n += 1
    for i in range(22):                                                      # neck / chest drape
        a = rnd.uniform(0, 2 * math.pi)
        rootp = (0.22 * math.cos(a), 0.80 + rnd.uniform(-0.10, 0.05), 0.02 + 0.24 * math.sin(a))
        outd = vnorm((math.cos(a), -0.3, math.sin(a)))
        v, f, uv = mane_cluster_geo(rnd, 14, rnd.uniform(0.35, 0.55), outd, (0.0, -0.45, -0.9), 1.0)
        L['mane'].append((mesh_object("ManeB%d" % i, v, f, uv, mane_m, True, chest, loc=rootp), n))
        n += 1
    # ---- legs ----
    L['legs'] = {}
    for nm, front, x, off, stand in (("LH", False, -0.17, 0.0, 0.04), ("LF", True, -0.19, 0.25, -0.03),
                                     ("RH", False, 0.17, 0.5, -0.04), ("RF", True, 0.19, 0.75, 0.03)):
        leg = build_leg("Lion" + nm, front, x, chest if front else root, SH_Y if front else HIP_Y, fur)
        leg['off'], leg['stand'] = off, stand
        L['legs'][nm] = leg
    # ---- tail ----
    tail = [empty("Tail0", (0, -0.90, 0.08), parent=root)]
    seg = 0.11
    for k in range(8):
        r0, r1 = 0.056 - k * 0.0032, 0.056 - (k + 1) * 0.0032
        v, f, uv = tube_geo([(0, 0, 0), (0, 0, -seg * 0.5), (0, 0, -seg * 1.08)], [r0, (r0 + r1) / 2, r1], 8, False, False)
        mesh_object("TailSeg%d" % k, v, f, uv, fur, True, tail[-1])
        tail.append(empty("Tail%d" % (k + 1), (0, 0, -seg), parent=tail[-1]))
    mb = MeshBuilder()
    mb.add(*ellipsoid_geo((0, 0, -0.09), (0.05, 0.05, 0.10)))
    for i in range(16):
        a = 2 * math.pi * i / 16
        d = vnorm((math.cos(a) * 0.5, math.sin(a) * 0.5, -1.0))
        pts = [(0, 0, -0.02), vadd((0, 0, -0.02), vmul(d, 0.09)), vadd((0, 0, -0.02), vmul(d, 0.18))]
        mb.add(*tube_geo(pts, [0.012, 0.009, 0.002], 4, False, True))
    mb.build("TailTuft", dark, True, tail[-1])
    L['tail'] = tail
    G['L'] = L


# ----------------------------------------------------------------------------
#  MOTION PLAN, CAMERA AND GAIT
# ----------------------------------------------------------------------------
LION_Y0, LION_Y1 = 5.1, 2.1
T_STOP = 5.4                      # the lion comes to a halt here
T_TOTAL = (F_END - F_START) / float(FPS)


def cam_pos(t):
    tn = clamp(t / T_TOTAL)
    e = tn * tn * (3.0 - 2.0 * tn)
    return (lerp(-1.1, 0.30, e) + 0.10 * math.sin(2 * math.pi * tn * 0.9),
            lerp(-2.2, -0.45, e),
            lerp(0.62, 0.95, e) + 0.025 * math.sin(2 * math.pi * tn * 1.3))


def lion_plan():
    N = 400
    ys = [LION_Y0 + (LION_Y1 - LION_Y0) * i / (N - 1.0) for i in range(N)]
    xs = [path_x(y) + 0.06 * math.sin(2.1 * y + 1.0) for y in ys]
    cum = [0.0]
    for i in range(1, N):
        cum.append(cum[-1] + math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]))
    ltot = cum[-1]

    def pos_at(s):
        s = clamp(s, 0.0, ltot)
        lo, hi = 0, N - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if cum[mid] <= s:
                lo = mid
            else:
                hi = mid
        u = (s - cum[lo]) / max(cum[hi] - cum[lo], 1e-9)
        return (lerp(xs[lo], xs[hi], u), lerp(ys[lo], ys[hi], u))

    def heading_at(s):
        a, b = pos_at(s - 0.15), pos_at(s + 0.15)
        return heading_to(b[0] - a[0], b[1] - a[1])
    dt = 1.0 / FPS
    raw = [0.0]
    for f in frames():
        t = tsec(f)
        raw.append(raw[-1] + (1.0 - smoothstep(2.3, T_STOP, t)) * dt)
    scale = ltot / raw[-1]
    S = [r * scale for r in raw[1:]]
    stride = ltot / (round(ltot / 0.8 * 4.0) / 4.0)
    V = []
    for i in range(len(S)):
        a, b = S[max(i - 1, 0)], S[min(i + 1, len(S) - 1)]
        V.append((b - a) * FPS / float(min(i + 1, len(S) - 1) - max(i - 1, 0)))
    return dict(S=S, PH=[s / stride for s in S], V=V, vpeak=max(V), stride=stride, pos_at=pos_at,
                heading_at=heading_at, E=stride * BETA)


def leg_pose(sp, hip_z, phase, E, settle, stand):
    if phase < BETA:
        u = phase / BETA
        dy, lift, sw = E * (0.5 - u), 0.0, 0.0
    else:
        u = (phase - BETA) / (1.0 - BETA)
        e = u * u * (3.0 - 2.0 * u)
        dy, lift, sw = E * (-0.5 + e), sp['lift'] * math.sin(math.pi * u), math.sin(math.pi * u)
    dy = lerp(dy, stand, settle)
    ty = sp['yc'] + dy
    tz = 0.08 + lift - hip_z
    psi = sp['psi0'] - sp['k'] * sw
    ay, az = ty - sp['l3'] * math.sin(psi), tz + sp['l3'] * math.cos(psi)
    a1, a2 = ik2_planar(sp['l1'], sp['l2'], ay, az, sp['bend'])
    a3 = psi - (a1 + a2)
    a4 = -0.45 * sw - psi
    return a1, a2, a3, a4


def bump_at(t, t0, dur):
    if t < t0 or t > t0 + dur:
        return 0.0
    return math.sin(math.pi * (t - t0) / dur)


def step_animate():
    L, cam, tgt, rig = G['L'], G['cam'], G['tgt'], G['rig']
    plan = lion_plan()
    root, chest, neck, head, jaw = L['root'], L['chest'], L['neck'], L['head'], L['jaw']
    E = plan['E']
    prev_yaw = prev_pitch = 0.0
    two_pi = 2.0 * math.pi
    for f in frames():
        i = f - F_START
        t = tsec(f)
        s, ph = plan['S'][i], plan['PH'][i]
        px, py = plan['pos_at'](s)
        h = plan['heading_at'](s)
        walk = clamp(plan['V'][i] / plan['vpeak'])
        bob = 0.011 * walk * math.cos(2 * two_pi * ph - 0.5)
        roll = 0.020 * walk * math.sin(two_pi * ph)
        pitch0 = 0.012 * walk * math.sin(2 * two_pi * ph + 0.9) + 0.004 * math.sin(two_pi * 0.33 * t)
        root.location = (px, py, Z0 + bob)
        root.rotation_euler = (pitch0, roll, h)
        key_trs(root, f)
        chest.rotation_euler = (0.0, -roll * 0.5, 0.030 * walk * math.sin(two_pi * ph + 0.6))
        chest.keyframe_insert(data_path='rotation_euler', frame=f)
        br = 1.0 + 0.012 * math.sin(two_pi * 0.42 * t)
        L['chest_mesh'].scale = (br, 1.0, br)
        L['chest_mesh'].keyframe_insert(data_path='scale', frame=f)
        # legs
        settle = smoothstep(T_STOP, T_STOP + 1.1, t)
        for leg in L['legs'].values():
            sp = leg['sp']
            a1, a2, a3, a4 = leg_pose(sp, Z0 + bob + sp['hz'], (ph - leg['off']) % 1.0, E, settle, leg['stand'])
            for jn, ang in (('hip', a1), ('knee', a2), ('ankle', a3), ('paw', a4)):
                leg[jn].rotation_euler = (ang, 0.0, 0.0)
                leg[jn].keyframe_insert(data_path='rotation_euler', frame=f)
        # gaze: scan the forest while walking, then lock eyes with the camera
        hw = vadd((px, py, 0.0), rot_xyz((0.0, 1.06, 0.0), (0.0, 0.0, h)))
        cp = cam_pos(t)
        gw = smoothstep(4.5, 5.9, t)
        yaw_cam = wrap_pi(heading_to(cp[0] - hw[0], cp[1] - hw[1]) - h)
        pitch_cam = math.atan2(cp[2] - (Z0 + 0.15), math.hypot(cp[0] - hw[0], cp[1] - hw[1]))
        yaw = lerp(0.16 * math.sin(0.8 * t + 0.5), yaw_cam, gw)
        pit = lerp(-0.12 + 0.04 * math.sin(1.1 * t), pitch_cam, gw) + 0.035 * walk * math.sin(2 * two_pi * ph + 1.3)
        neck.rotation_euler = (pit * 0.45, 0.0, yaw * 0.45)
        head.rotation_euler = (pit * 0.55, 0.0, yaw * 0.55)
        neck.keyframe_insert(data_path='rotation_euler', frame=f)
        head.keyframe_insert(data_path='rotation_euler', frame=f)
        dyaw = (yaw - prev_yaw) * FPS * 0.10
        dpit = (pit - prev_pitch) * FPS * 0.10
        prev_yaw, prev_pitch = yaw, pit
        jaw.rotation_euler = (0.02 + 0.015 * math.sin(two_pi * 0.4 * t), 0.0, 0.0)
        jaw.keyframe_insert(data_path='rotation_euler', frame=f)
        # blink (lids slide down over the eyes) + ear flicks
        for lid in L['lids']:
            z_open = 0.078
            lid.location = (lid.location[0], lid.location[1],
                            z_open - 0.040 * max(bump_at(t, 2.6, 0.22), bump_at(t, 6.15, 0.26)))
            lid.keyframe_insert(data_path='location', frame=f)
        for ej, sd in L['ears']:
            fl = max(bump_at(t, 1.2, 0.2), bump_at(t, 3.4, 0.22), bump_at(t, 5.0, 0.2)) * (0.30 if sd < 0 else 0.0)
            fl += max(bump_at(t, 2.1, 0.2), bump_at(t, 4.2, 0.2)) * (0.30 if sd > 0 else 0.0)
            ej.rotation_euler = (-0.12 * fl, 0.0, sd * 0.9 * fl)
            ej.keyframe_insert(data_path='rotation_euler', frame=f)
        # tail
        for k, tj in enumerate(L['tail']):
            base = -0.45 if k == 0 else 0.075
            sway = 0.12 * walk * math.sin(two_pi * ph - 0.6 * k) + 0.05 * math.sin(1.2 * t - 0.45 * k)
            tj.rotation_euler = (base + 0.06 * math.sin(1.0 * t - 0.3 * k + 1.0), sway, 0.0)
            tj.keyframe_insert(data_path='rotation_euler', frame=f)
        # mane: wind + gait bounce + lag behind head turns
        for ob, idx in L['mane']:
            p_i = 0.37 * idx
            wind = 0.06 * math.sin(1.9 * t + p_i)
            gait = 0.07 * walk * math.sin(2 * two_pi * ph + 1.0 - 0.18 * (idx % 7))
            ob.rotation_euler = (wind + gait - 0.6 * dpit, 0.02 * math.sin(1.3 * t + p_i),
                                 -0.8 * dyaw + 0.04 * math.cos(1.5 * t + p_i))
            ob.keyframe_insert(data_path='rotation_euler', frame=f)
        # camera, focus target and light rig follow the action
        target = (lerp(px, hw[0], smoothstep(0.0, 1.0, t / T_TOTAL)), lerp(py, hw[1], smoothstep(0.0, 1.0, t / T_TOTAL)),
                  lerp(0.66, 1.03, smoothstep(0.0, 1.0, t / T_TOTAL)))
        tgt.location = target
        tgt.keyframe_insert(data_path='location', frame=f)
        cam.location = cp
        cam.keyframe_insert(data_path='location', frame=f)
        rig.location = (px, py, 0.0)
        rig.keyframe_insert(data_path='location', frame=f)


# ----------------------------------------------------------------------------
#  CAMERA, LIGHTS, FX
# ----------------------------------------------------------------------------
def step_camera():
    tgt = empty("CamTarget", (0.7, 4.0, 0.7), size=0.2)
    G['tgt'] = tgt
    G['cam'] = add_camera(40, cam_pos(0.0), tgt, focus=tgt, fstop=3.2)


def step_lights():
    rig = empty("LightRig", (0, 4, 0))
    G['rig'] = rig
    add_light('AREA', 'KeyFill', (-2.0, -3.2, 2.6), 380.0, (1.0, 0.90, 0.78), size=3.0, direction=(2.0, 3.2, -1.6), parent=rig)
    add_light('SPOT', 'RimLeft', (-1.7, 2.6, 2.4), 900.0, (1.0, 0.80, 0.55), size=0.25, direction=(1.7, -2.6, -1.4),
              spot=1.0, parent=rig)
    add_light('SPOT', 'RimRight', (1.7, 2.6, 2.4), 900.0, (1.0, 0.80, 0.55), size=0.25, direction=(-1.7, -2.6, -1.4),
              spot=1.0, parent=rig)


def step_fx():
    cam = G['cam']
    make_motes("DustA", 240, (-4, -1.5, 0.2), (4, 9, 3.4), 0.004, 0.010, (1.0, 0.88, 0.6), 7.0, 1, (0.12, 0.05, 0.03))
    make_motes("DustB", 220, (-6, -1.5, 0.3), (6, 14, 5.0), 0.006, 0.014, (1.0, 0.9, 0.7), 5.0, 2, (-0.10, 0.04, 0.04))
    make_motes("DustC", 160, (-3, -1, 0.2), (3, 6, 2.5), 0.008, 0.018, (1.0, 0.85, 0.55), 4.0, 3, (0.05, -0.03, 0.05))
    make_shafts(SUN_DIR, cam, 9, (-7, 7), (3, 24), 18.0, (0.7, 2.0), (1.0, 0.84, 0.55), 0.35, 4, avoid=near_path(0.6))


def main():
    clear_scene()
    scene = bpy.context.scene
    random.seed(11)
    run_steps([
        ("world, sun and haze", step_world),
        ("ground", step_ground),
        ("trees", step_trees),
        ("undergrowth", step_undergrowth),
        ("camera", step_camera),
        ("lion", step_lion),
        ("lights", step_lights),
        ("dust and light shafts", step_fx),
        ("animation", step_animate),
    ])
    setup_render(scene, OUTPUT_NAME, exposure=0.25, volumetrics=USE_VOLUMETRICS)
    add_glare(scene)
    finish(scene)


main()
