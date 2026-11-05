"""Build the 18-DoF Go1 + 6-DoF arm MJCF used by the RoboDuet-on-MuJoCo prototype.

Input : models/vendor/go1_menagerie.xml  (built-in Go1 from MuJoCo Menagerie:
        real inertials, joint limits, foot contact params)
Output: models/go1_arm.xml               (18 actuated DoF, no mesh dependency)

The arm is a 6-DoF stand-in, not the ARX5 URDF: same DoF count and same
default pose [0, 0.8, 0.8, 0, 0, 0] as the paper, but it is NOT dimensionally
faithful -- measured 4.25 kg vs ARX5's 3.35 kg, and 0.792 m base->tip vs the
ARX5's 620 mm. Swapping in the real URDF only changes the <body> chain below.
"""
import re
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / "vendor" / "go1_menagerie.xml"
DST = HERE / "go1_arm.xml"

# ----------------------------------------------------------------------------
# 6-DoF arm.  Segments 0.05+0.27+0.26+0.05+0.05+0.072 give a 0.792 m base->tip
# chain (ARX5 spec is 620 mm, so this stand-in reaches ~28% further).
# Masses sum to 4.25 kg (ARX5 + gripper in the paper is 3.35 kg).
# Default pose [0, 0.8, 0.8, 0, 0, 0] reaches forward and slightly up.
# ----------------------------------------------------------------------------
ARM_XML = """
      <!-- ================= 6-DoF arm (ARX5-sized) ================= -->
      <body name="arm_base" pos="0.0 0.0 0.055">
        <geom class="arm" type="cylinder" size="0.045 0.02" pos="0 0 0.02"/>
        <inertial pos="0 0 0.02" mass="0.5" diaginertia="0.0008 0.0008 0.0010"/>
        <body name="arm_link1" pos="0 0 0.04">
          <joint name="arm_j1" class="armj" axis="0 0 1" range="-2.90 2.90"/>
          <inertial pos="0 0 0.02" mass="0.9" diaginertia="0.0020 0.0020 0.0009"/>
          <geom class="arm" type="capsule" fromto="0 0 0 0 0 0.05" size="0.033"/>
          <body name="arm_link2" pos="0 0 0.05">
            <joint name="arm_j2" class="armj" axis="0 1 0" range="-2.60 2.60"/>
            <inertial pos="0 0 0.135" mass="0.9" diaginertia="0.0060 0.0060 0.0009"/>
            <geom class="arm" type="capsule" fromto="0 0 0 0 0 0.27" size="0.030"/>
            <body name="arm_link3" pos="0 0 0.27">
              <joint name="arm_j3" class="armj" axis="0 1 0" range="-2.90 2.90"/>
              <inertial pos="0 0 0.13" mass="0.8" diaginertia="0.0045 0.0045 0.0007"/>
              <geom class="arm" type="capsule" fromto="0 0 0 0 0 0.26" size="0.024"/>
              <body name="arm_link4" pos="0 0 0.26">
                <joint name="arm_j4" class="armj" axis="0 0 1" range="-3.00 3.00"/>
                <inertial pos="0 0 0.02" mass="0.5" diaginertia="0.0006 0.0006 0.0005"/>
                <geom class="arm" type="capsule" fromto="0 0 0 0 0 0.05" size="0.020"/>
                <body name="arm_link5" pos="0 0 0.05">
                  <joint name="arm_j5" class="armj" axis="0 1 0" range="-2.20 2.20"/>
                  <inertial pos="0 0 0.02" mass="0.35" diaginertia="0.0003 0.0003 0.0002"/>
                  <geom class="arm" type="capsule" fromto="0 0 0 0 0 0.05" size="0.016"/>
                  <body name="arm_link6" pos="0 0 0.05">
                    <joint name="arm_j6" class="armj" axis="0 0 1" range="-3.00 3.00"/>
                    <inertial pos="0 0 0.02" mass="0.30" diaginertia="0.0002 0.0002 0.0002"/>
                    <geom class="arm" type="box" size="0.016 0.016 0.018" pos="0 0 0.018"/>
                    <geom class="gripper" type="box" size="0.010 0.040 0.018" pos="0 0 0.050"/>
                    <site name="ee" pos="0 0 0.072" size="0.008" rgba="1 0 0 1" group="5"/>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
      <!-- ======================= end arm ======================== -->
"""

FLOOR = """
    <geom name="floor" type="plane" size="0 0 0.05" contype="1" conaffinity="1"
          rgba="0.45 0.47 0.5 1" friction="1.0 0.005 0.0001" condim="3"/>
"""


def main():
    xml = SRC.read_text(encoding="utf-8")

    # 1) drop mesh assets and all visual geoms -> no STL dependency, faster contacts
    xml = re.sub(r'^\s*<mesh [^/]*/>\s*\n', '', xml, flags=re.M)
    xml = re.sub(r'^\s*<geom class="visual"[^/]*/>\s*\n', '', xml, flags=re.M)
    xml = xml.replace(' meshdir="assets"', '')

    # 1b) the collision geoms were in visibility group 3 (hidden), which was
    #     fine when the STL visuals existed. With the meshes stripped they are
    #     the only geometry, so make them renderable and give them a colour.
    xml = xml.replace('<geom group="3" type="capsule"/>',
                      '<geom type="capsule" rgba="0.42 0.44 0.50 1"/>')
    xml = xml.replace('<geom type="sphere" size="0.023" pos="0 0 -0.213" '
                      'priority="1"',
                      '<geom type="sphere" size="0.023" pos="0 0 -0.213" '
                      'rgba="0.13 0.13 0.15 1" priority="1"')

    # 2) joint-level passive params for the arm + geom classes
    arm_defaults = """
      <default class="armj">
        <joint axis="0 0 1" damping="1.5" armature="0.01" frictionloss="0.1"/>
        <position kp="40" kv="1.0" forcerange="-18 18"/>
      </default>
      <default class="arm">
        <geom friction="0.5" condim="3" contype="1" conaffinity="1" rgba="0.55 0.55 0.58 1"/>
      </default>
      <default class="gripper">
        <geom friction="0.8" condim="3" contype="1" conaffinity="1" rgba="0.15 0.15 0.18 1"/>
      </default>
    """
    # MuJoCo >= 3.3 rejects extra *top-level* <default class=...> blocks, so the
    # arm classes are nested inside the single top-level default block.
    xml = xml.replace('  <default>\n', '  <default>\n' + arm_defaults, 1)

    # 3) Go1 leg actuators: stiff-ish position servo instead of the Menagerie
    #    kp=100 (undamped). Gains were picked by a sweep on this 17 kg model
    #    (see README): kp=40/kv=1.0 gives a 0.239 m stand with 0.18 deg roll,
    #    3.6 deg pitch and <0.1 mm height jitter. The paper's exact gains are
    #    not published; legged_gym's 20/0.5 sags to 0.186 m on this mass.
    xml = xml.replace('<position kp="100" forcerange="-23.7 23.7"/>',
                      '<position kp="40" kv="1.0" forcerange="-23.7 23.7"/>')

    # 4) sim options: 200 Hz physics, 4 substeps per 50 Hz control step
    xml = xml.replace('<option cone="elliptic" impratio="100"/>',
                      '<option timestep="0.005" cone="elliptic" impratio="100" '
                      'iterations="20" ls_iterations="10" '
                      'integrator="implicitfast" solver="Newton"/>')

    # 5) floor + arm. The arm body chain is appended *after* the four legs so
    #    that qpos order is the legged_gym convention: qpos[7:19] legs,
    #    qpos[19:25] arm. Inserting it before the legs silently shifts every
    #    keyframe, actuator and observation slice by six entries.
    xml = xml.replace('  <worldbody>\n', '  <worldbody>\n' + FLOOR, 1)
    anchor = '\n    </body>\n  </worldbody>'
    assert xml.count(anchor) == 1, "trunk closing tag not found"
    xml = xml.replace(anchor, ARM_XML + anchor, 1)

    # 6) collect arm actuators from the joints we just wrote
    arm_joints = re.findall(r'<joint name="(arm_j\d)"', ARM_XML)
    acts = "\n".join(
        f'    <position class="armj" name="{j}" joint="{j}"/>' for j in arm_joints)
    xml = xml.replace('  </actuator>', acts + '\n  </actuator>')

    # 7) home keyframe: legs from Menagerie, arm at the paper's default pose
    legs = "0 0.9 -1.8 0 0.9 -1.8 0 0.9 -1.8 0 0.9 -1.8"
    arm = "0 0.8 0.8 0 0 0"
    xml = re.sub(r'  <keyframe>.*?</keyframe>',
                 f'  <keyframe>\n    <key name="home" qpos="0 0 0.29 1 0 0 0 {legs} {arm}"\n'
                 f'      ctrl="{legs} {arm}"/>\n  </keyframe>', xml, flags=re.S)

    DST.write_text(xml, encoding="utf-8")
    print(f"wrote {DST}  ({len(xml)} bytes)")


if __name__ == "__main__":
    main()
