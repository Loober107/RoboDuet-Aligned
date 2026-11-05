"""Step 2: generate an MJCF for the paper's real robot from arx5p2Go1.urdf.

Everything that can be taken from the official files is taken from them:
  * the body tree, joint origins/axes/limits      <- URDF
  * mass, COM, full inertia tensor                <- URDF
  * collision geometry                            <- URDF <collision> only
  * default joint angles                          <- go1_gym/envs/go1/asset_config.py
  * PD gains (effective, see note below)          <- same file
  * sim dt / decimation / effort limits           <- legged_robot_config.py, URDF

Deliberate deviations, all forced by MuJoCo:
  * collision meshes are convex hulls (MuJoCo cannot do concave trimesh).
  * parent/child collision pairs stay disabled (MuJoCo default); Isaac Gym with
    self_collisions=0 enables them.
  * the two gripper joints get no actuator (matching p_gains=0 for i>=num_actions).

Note on arm gains: asset_config.py lists per-joint arm gains, but
legged_robot.py:1336 only looks up names present in Cfg.control.stiffness
({'joint','widow','zarx','zarx_j3'}), so zarx_j1/j2/j4/j5/j6 all resolve to
stiffness_arm['zarx']=50 / damping 20 and only j3 hits its own entry (70/15).
This script reproduces the *effective* values, not the dead dict entries.

Run:  python models/build_mjcf.py
"""
import math
import os
import pathlib
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
URDF = ROOT / "vendor" / "roboduet_official" / "resources" / "robots" / "arx5p2Go1" / "urdf" / "arx5p2Go1.urdf"
DST = HERE / "arx5p2Go1.xml"
ROOT_Z = 0.32

# ---------------------------------------------------------------- official values
DEFAULT_ANGLES = {
    "FL_hip_joint": 0.1, "RL_hip_joint": 0.1,
    "FR_hip_joint": -0.1, "RR_hip_joint": -0.1,
    "FL_thigh_joint": 0.8, "FR_thigh_joint": 0.8,
    "RL_thigh_joint": 1.0, "RR_thigh_joint": 1.0,
    "FL_calf_joint": -1.5, "FR_calf_joint": -1.5,
    "RL_calf_joint": -1.5, "RR_calf_joint": -1.5,
    "zarx_j1": 0.0, "zarx_j2": 0.8, "zarx_j3": 0.8,
    "zarx_j4": 0.0, "zarx_j5": 0.0, "zarx_j6": 0.0,
    "zarx_j7": 0.0, "zarx_j8": 0.0,
}
LEG_KP, LEG_KD = 35.0, 1.0
ARM_GAIN = {"zarx_j3": (70.0, 15.0)}      # everything else falls through to 'zarx'
ARM_GAIN_DEFAULT = (50.0, 20.0)
NUM_ACTIONS = 18                            # 12 legs + 6 arm; grippers are passive

MESH_FILES = {"trunk": "trunk.stl", "base_link": "base_link.stl",
              **{f"link{i}": f"link{i}.stl" for i in range(1, 9)}}
MESH_LINK = {"trunk": "trunk", "base_link": "base_link",
             **{f"zarx_body{i}": f"link{i}" for i in range(1, 9)}}


def rpy_to_quat(r, p, y):
    """URDF fixed-axis roll-pitch-yaw (R = Rz*Ry*Rx) -> MuJoCo wxyz quat."""
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return [w, x, y, z]


def fvec(s, default=(0.0, 0.0, 0.0)):
    return [float(v) for v in s.split()] if s else list(default)


def fmt(v):
    return " ".join(f"{x:.9g}" for x in v)


class Gen:
    def __init__(self):
        self.root = ET.parse(URDF).getroot()
        self.links = {}
        for l in self.root.findall("link"):
            inert = l.find("inertial")
            rec = dict(name=l.get("name"), mass=0.0, com=[0, 0, 0], I=None,
                       cols=[])
            if inert is not None:
                m = inert.find("mass")
                rec["mass"] = float(m.get("value")) if m is not None else 0.0
                o = inert.find("origin")
                if o is not None:
                    rec["com"] = fvec(o.get("xyz"))
                i = inert.find("inertia")
                if i is not None:
                    rec["I"] = [float(i.get(k, 0)) for k in
                                ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")]
            for c in l.findall("collision"):
                g = c.find("geometry")
                o = c.find("origin")
                geo = {"pos": fvec(o.get("xyz")) if o is not None else [0, 0, 0],
                       "quat": rpy_to_quat(*fvec(o.get("rpy"))) if o is not None
                       else [1, 0, 0, 0]}
                for ch in list(g):
                    geo["kind"] = ch.tag
                    if ch.tag == "box":
                        geo["size"] = [float(v) / 2 for v in ch.get("size").split()]
                    elif ch.tag == "sphere":
                        geo["size"] = [float(ch.get("radius"))]
                    elif ch.tag == "mesh":
                        geo["file"] = ch.get("filename").split("/")[-1]
                    rec["cols"].append(geo)
            self.links[rec["name"]] = rec

        self.joints = []
        self.children = {}
        for j in self.root.findall("joint"):
            o = j.find("origin")
            ax = j.find("axis")
            lim = j.find("limit")
            rec = dict(name=j.get("name"), type=j.get("type"),
                       parent=j.find("parent").get("link"),
                       child=j.find("child").get("link"),
                       pos=fvec(o.get("xyz")) if o is not None else [0, 0, 0],
                       quat=rpy_to_quat(*fvec(o.get("rpy"))) if o is not None
                       else [1, 0, 0, 0],
                       axis=fvec(ax.get("xyz")) if ax is not None else [0, 0, 1],
                       lower=float(lim.get("lower")) if lim is not None else None,
                       upper=float(lim.get("upper")) if lim is not None else None,
                       effort=float(lim.get("effort")) if lim is not None else None)
            self.joints.append(rec)
            self.children.setdefault(rec["parent"], []).append(rec)

    def emit_body(self, name, joint, depth, out, dof_counter, act_lines):
        L = self.links[name]
        pad = "  " * depth
        if joint is None:                      # root body (trunk)
            out.append(f'{pad}<body name="{name}" pos="0 0 {ROOT_Z}">')
            out.append(f'{pad}  <freejoint/>')
        else:
            attrs = [f'pos="{fmt(joint["pos"])}"']
            if joint["quat"] != [1, 0, 0, 0]:
                attrs.append(f'quat="{fmt(joint["quat"])}"')
            out.append(f'{pad}<body name="{name}" {" ".join(attrs)}>')

        if joint is not None and joint["type"] != "fixed":
            jn = joint["name"]
            if joint["type"] == "revolute":
                jt = "hinge"
            elif joint["type"] == "prismatic":
                jt = "slide"
            else:
                jt = "hinge"
            lim = ""
            if joint["lower"] is not None and joint["lower"] != joint["upper"]:
                lim = f' range="{joint["lower"]:.10g} {joint["upper"]:.10g}"'
            out.append(f'{pad}  <joint name="{jn}" type="{jt}" '
                       f'axis="{fmt(joint["axis"])}"{lim}/>')

        # ---- inertial
        if L["mass"] > 0 and L["I"] is not None:
            out.append(f'{pad}  <inertial pos="{fmt(L["com"])}" '
                       f'mass="{L["mass"]:.10g}" fullinertia="{fmt(L["I"])}"/>')

        # ---- collision geoms (visuals deliberately omitted)
        for c in L["cols"]:
            p = f'pos="{fmt(c["pos"])}"'
            q = "" if c["quat"] == [1, 0, 0, 0] else f' quat="{fmt(c["quat"])}"'
            if c["kind"] == "mesh":
                key = MESH_LINK.get(name)
                if key is None:
                    continue
                out.append(f'{pad}  <geom class="colmesh" type="mesh" '
                           f'mesh="{key}" {p}{q}/>')
            elif c["kind"] == "box":
                out.append(f'{pad}  <geom class="col" type="box" '
                           f'size="{fmt(c["size"])}" {p}{q}/>')
            elif c["kind"] == "sphere":
                out.append(f'{pad}  <geom class="col" type="sphere" '
                           f'size="{fmt(c["size"])}" {p}{q}/>')

        kids = list(self.children.get(name, []))
        if joint is None:
            # URDF root is the massless frame `base`, which coincides with
            # `trunk` (floating_base origin = 0 0 0). The arm hangs off `base`,
            # so pull its chain over to the trunk to keep one root body.
            kids += [j for j in self.children.get("base", [])
                     if j["child"] != "trunk"]

        for j in kids:
            child = j["child"]
            if j["type"] == "fixed":
                self.emit_body(child, j, depth + 1, out, dof_counter, act_lines)
                continue
            dof_counter.append(j)
            if len(dof_counter) <= NUM_ACTIONS:
                if joint_is_leg(j["name"]):
                    act_lines.append(
                        f'    <motor name="{j["name"]}" joint="{j["name"]}" '
                        f'ctrlrange="{-j["effort"]:.10g} {j["effort"]:.10g}" '
                        f'forcerange="{-j["effort"]:.10g} {j["effort"]:.10g}"/>')
                else:
                    kp, kd = ARM_GAIN.get(j["name"], ARM_GAIN_DEFAULT)
                    act_lines.append(
                        f'    <position name="{j["name"]}" joint="{j["name"]}" '
                        f'kp="{kp:.10g}" kv="{kd:.10g}" '
                        f'ctrlrange="{j["lower"]:.10g} {j["upper"]:.10g}" '
                        f'forcerange="{-j["effort"]:.10g} {j["effort"]:.10g}"/>')
            self.emit_body(child, j, depth + 1, out, dof_counter, act_lines)

        out.append(f"{pad}</body>")

    def build(self):
        out, dofs, acts = [], [], []
        out.append('<mujoco model="arx5p2Go1_real">')
        out.append('  <compiler angle="radian" meshdir="meshes" autolimits="true"/>')
        out.append('  <option timestep="0.005" integrator="implicitfast" '
                   'solver="Newton" iterations="20" ls_iterations="10" '
                   'cone="pyramidal"/>')
        out.append('  <default>')
        out.append('    <default class="col">')
        out.append('      <geom friction="1.0" condim="3" contype="1" '
                   'conaffinity="1" rgba="0.42 0.44 0.50 1"/>')
        out.append('    </default>')
        out.append('    <default class="colmesh">')
        out.append('      <geom friction="1.0" condim="3" contype="1" '
                   'conaffinity="1" rgba="0.55 0.56 0.60 1"/>')
        out.append('    </default>')
        out.append('  </default>')
        out.append('  <asset>')
        for key, f in MESH_FILES.items():
            out.append(f'    <mesh name="{key}" file="{f}"/>')
        out.append('  </asset>')
        out.append('  <worldbody>')
        out.append('    <geom name="floor" type="plane" size="0 0 0.05" '
                   'rgba="0.45 0.47 0.5 1" friction="1.0 0.005 0.0001" condim="3"/>')
        # root: trunk (base is a massless frame at the same origin)
        self.emit_body("trunk", None, 2, out, dofs, acts)
        out.append('  </worldbody>')

        xml = "\n".join(out)
        xml += "\n  <actuator>\n" + "\n".join(acts) + "\n  </actuator>\n"
        xml += keyframe(dofs)
        xml += "\n</mujoco>\n"
        return xml


def joint_is_leg(name):
    return name.startswith(("FR_", "FL_", "RR_", "RL_"))


def keyframe(dofs):
    q = [DEFAULT_ANGLES.get(j["name"], 0.0) for j in dofs]
    return ('  <keyframe>\n    <key name="home" '
            f'qpos="0 0 {ROOT_Z} 1 0 0 0 {" ".join(f"{v:.10g}" for v in q)}"\n'
            f'      ctrl="{" ".join("0" for _ in range(NUM_ACTIONS))}"/>\n'
            '  </keyframe>')


def lowest_point(m, d):
    """Lowest world-z over all non-world geoms.

    Do NOT use m.geom_aabb: it is not centred on the geom frame (for a sphere it
    comes back as [0,0,0]..[2r,2r,2r]), which silently places the robot ~18 mm
    too low and starts the sim with the feet inside the floor. Compute the real
    support point per geom type instead.
    """
    lo = 1e9
    R = lambda gi: d.geom_xmat[gi].reshape(3, 3)
    for gi in range(m.ngeom):
        if m.geom_bodyid[gi] == 0:
            continue
        t = m.geom_type[gi]
        s = m.geom_size[gi]
        if t == mujoco.mjtGeom.mjGEOM_SPHERE:
            lo = min(lo, float(d.geom_xpos[gi][2] - s[0]))
        elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
            lo = min(lo, float(d.geom_xpos[gi][2] - s[0] - s[1]))
        elif t == mujoco.mjtGeom.mjGEOM_BOX:
            sx, sy, sz = s[0], s[1], s[2]
            c = np.array([[sx * ex, sy * ey, sz * ez]
                          for ex in (-1, 1) for ey in (-1, 1) for ez in (-1, 1)])
            world = c @ R(gi).T + d.geom_xpos[gi]
            lo = min(lo, float(world[:, 2].min()))
        elif t == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[gi]
            a = m.mesh_vertadr[mid]
            v = m.mesh_vert[a:a + m.mesh_vertnum[mid]]
            world = v @ R(gi).T + d.geom_xpos[gi]
            lo = min(lo, float(world[:, 2].min()))
    return lo


def main():
    g = Gen()
    xml = g.build()
    os.chdir(HERE)          # so meshdir="meshes" resolves for from_xml_string
    m = mujoco.MjModel.from_xml_string(xml)
    print(f"compiled: nq={m.nq} nv={m.nv} nu={m.nu} nbody={m.nbody} "
          f"ngeom={m.ngeom} mass={m.body_mass.sum():.3f} kg")

    # place the feet on the floor: measure the lowest point at the home pose
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    lo = lowest_point(m, d)
    new_z = ROOT_Z - lo + 0.002
    xml = xml.replace(f'pos="0 0 {ROOT_Z}"', f'pos="0 0 {new_z:.4f}"')
    xml = xml.replace(f'qpos="0 0 {ROOT_Z}', f'qpos="0 0 {new_z:.4f}')
    DST.write_text(xml, encoding="utf-8")
    print(f"lowest point at home: z={lo:.4f} at base {ROOT_Z} "
          f"-> base height {new_z:.4f} m")
    print(f"wrote {DST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
