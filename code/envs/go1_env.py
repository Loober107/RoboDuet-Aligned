"""Stage-1 vectorised environment for the RoboDuet reproduction (MuJoCo, CPU).

Follows the official implementation:
  * control      : torque PD recomputed every 200 Hz substep, clipped to the
                   URDF effort limit              (legged_robot.py:180-190, 217-220)
  * action       : q_target = default + 0.25 * a (offsets scaled first), hips
                   scaled by 0.5                  (legged_robot.py:173-177,
                                                   go1_config.py:34-35)
  * stage-1 arm  : hard-locked every substep      (legged_robot.py:234-245)
  * observation  : 63 dims, the exact concatenation of compute_observations
                   (legged_robot.py:385-418)
  * commands     : Table II of the paper

Only the physics stepping runs on the thread pool; the reward is computed for
all envs at once with vectorised numpy, because the per-env scalar version cost
265 ms per control step for 1024 envs (measured) and dominated everything.
"""
import numpy as np
import mujoco

# ---------------------------------------------------------------- constants
DEFAULT_ANGLES = np.array([          # qpos order: FR, FL, RR, RL
    -0.1, 0.8, -1.5,     # FR  (asset_config.py:54-68)
     0.1, 0.8, -1.5,     # FL
    -0.1, 1.0, -1.5,     # RR
     0.1, 1.0, -1.5,     # RL
], dtype=np.float64)
ARM_DEFAULT = np.array([0.0, 0.8, 0.8, 0.0, 0.0, 0.0])
GRIP_DEFAULT = np.zeros(2)

KP, KD, TAU_MAX = 35.0, 1.0, 23.7
ACTION_SCALE = 0.25
HIP_SCALE = 0.5
HIP_IDX = np.array([0, 3, 6, 9])
DECIM = 4
DT = 0.005
CTRL_DT = DT * DECIM

LEGS = slice(7, 19)
ARM = slice(19, 25)
GRIP = slice(25, 27)
DOF_LEGS = slice(6, 18)
DOF_ARM = slice(18, 24)

CMD_SCALE = np.array([2.0, 2.0, 0.25])
DOF_POS_SCALE, DOF_VEL_SCALE = 1.0, 0.05

NUM_OBS = 63
HIST_LEN = 30              # wtw_config.py:66 num_observation_history
NUM_ACT = 12
EPISODE_S = 20.0
NUM_CTRL_STEPS = int(round(EPISODE_S / CTRL_DT))
SIGMA_REW_NEG = 5.0         # legged_robot_config.py:358 (wtw overrides to 0.02, which zeroes any reward)

# Paper SIII-B-1: "When the base velocity is zero ... we set clock time t_t to
# [1,1,1,1] to force all feet to maintain a stationary position."
ZERO_CMD_EPS = 0.05         # |vx|,|vy|,|wz| all below this => stand command
STAND_PHASE = 0.25          # sin(2*pi*0.25) = 1  =>  clock = [1,1,1,1]
STAND_PROB = 0.15           # fraction of resets that get a stand command


def rpy_from_quat(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    return roll, pitch


def quat_to_matrix(q):
    """(N,4) wxyz -> (N,3,3)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    m = np.empty((q.shape[0], 3, 3))
    m[:, 0, 0] = 1 - 2 * (y * y + z * z)
    m[:, 0, 1] = 2 * (x * y - w * z)
    m[:, 0, 2] = 2 * (x * z + w * y)
    m[:, 1, 0] = 2 * (x * y + w * z)
    m[:, 1, 1] = 1 - 2 * (x * x + z * z)
    m[:, 1, 2] = 2 * (y * z - w * x)
    m[:, 2, 0] = 2 * (x * z - w * y)
    m[:, 2, 1] = 2 * (y * z + w * x)
    m[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return m


class Go1Env:
    """Vectorised stage-1 env. Actions are (N, 12) joint offsets."""

    def __init__(self, model, num_envs, num_threads=8, seed=0, episode_steps=None):
        self.m = model
        self.n = int(num_envs)
        self.rng = np.random.default_rng(seed)
        self.episode_steps = episode_steps or NUM_CTRL_STEPS
        self.num_threads = max(1, int(num_threads))
        self.nq, self.nv = model.nq, model.nv

        self.data = [mujoco.MjData(model) for _ in range(self.n)]
        for d in self.data:
            mujoco.mj_resetDataKeyframe(model, d, 0)

        self.foot_bodies = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, s)
                            for s in ("FR_foot", "FL_foot", "RR_foot", "RL_foot")]
        assert min(self.foot_bodies) >= 0, "foot bodies not found"

        self.cmd = np.zeros((self.n, 3))
        # body attitude command [pitch, roll], official range +-0.4 rad
        self.body_cmd = np.zeros((self.n, 2))
        self.actions = np.zeros((self.n, 12))
        self.prev_actions = np.zeros((self.n, 12))
        self.gait_phase = np.zeros(self.n)
        self.gait_freq = np.full(self.n, 3.0)
        self.step_count = np.zeros(self.n, dtype=np.int64)
        self.base_lin_vel = np.zeros((self.n, 3))
        self.base_ang_vel = np.zeros((self.n, 3))
        self.prev_foot_xyz = None
        self.standing = np.zeros(self.n, dtype=bool)
        self._fw3 = np.zeros((self.n, 4, 3))
        # one model per env; domain randomisation replaces these with per-env
        # copies that share the constant geometry arrays (see go1_env_dr.py)
        # observation history, official HistoryWrapper: 30 frames of NUM_OBS
        self.obs_hist = np.zeros((self.n, HIST_LEN * NUM_OBS))
        self.models = [model] * self.n
        self.motor_strength = np.ones((self.n, 12))
        self.motor_offset = np.zeros((self.n, 12))
        self.last_contacts = np.zeros((self.n, 4), dtype=bool)
        self._thread_pool = None

    # ------------------------------------------------------------- helpers
    def _pool(self):
        if self._thread_pool is None:
            from concurrent.futures import ThreadPoolExecutor
            self._thread_pool = ThreadPoolExecutor(max_workers=self.num_threads)
        return self._thread_pool

    def _chunks(self):
        e = np.linspace(0, self.n, self.num_threads + 1).astype(int)
        return [(e[i], e[i + 1]) for i in range(self.num_threads)
                if e[i + 1] > e[i]]

    @staticmethod
    def _yaw_quat(yaw):
        return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])

    def _gather(self):
        qpos = np.empty((self.n, self.nq))
        qvel = np.empty((self.n, self.nv))
        for i, d in enumerate(self.data):
            qpos[i] = d.qpos
            qvel[i] = d.qvel
        return qpos, qvel

    # ---------------------------------------------------------------- reset
    def reset(self, idx=None):
        idx = np.arange(self.n) if idx is None else np.atleast_1d(idx)
        rng = self.rng
        for i in idx:
            d = self.data[i]
            mujoco.mj_resetDataKeyframe(self.m, d, 0)
            d.qpos[2] += rng.uniform(-0.02, 0.02)
            d.qpos[3:7] = self._yaw_quat(rng.uniform(-np.pi, np.pi))
            d.qvel[:] = 0.0
            d.qvel[0:3] = rng.normal(0, 0.3, 3)
            d.qvel[3:6] = rng.normal(0, 0.3, 3)
            d.qpos[LEGS] = DEFAULT_ANGLES + rng.normal(0, 0.08, 12)
            d.qpos[ARM] = ARM_DEFAULT
            d.qpos[GRIP] = GRIP_DEFAULT
            mujoco.mj_forward(self.m, d)

            # a fraction of resets get a "stand" command, otherwise the policy
            # would almost never see the zero-velocity regime and could not
            # learn the frozen-clock behaviour
            if rng.random() < STAND_PROB:
                self.cmd[i] = [0.0, 0.0, 0.0]
            else:
                self.cmd[i] = [rng.uniform(-1.0, 1.0), rng.uniform(-0.6, 0.6),
                               rng.uniform(-0.6, 0.6)]
            self.body_cmd[i] = [rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4)]
            self.actions[i] = 0.0
            self.prev_actions[i] = 0.0
            self.gait_phase[i] = rng.uniform(0, 1)
            self.gait_freq[i] = rng.uniform(2.5, 3.5)
            self.step_count[i] = 0
            self.base_lin_vel[i] = 0.0
            self.base_ang_vel[i] = 0.0
        self.prev_foot_xyz = None
        self.obs_hist[idx] = 0.0
        return self.obs()

    # ----------------------------------------------------------------- obs
    def clock(self):
        """Eq. (2)-(3) with trot offsets [0.5, 0, 0]: FR/RL in phase, FL/RR in phase.

        Standing envs output [1, 1, 1, 1] instead, per paper SIII-B-1, which
        freezes all four feet and stops the marching-in-place jitter.
        """
        t = self.gait_phase
        ph = np.stack([t + 0.0, t + 0.5, t + 0.5, t + 0.0], axis=1)
        out = np.sin(2 * np.pi * ph)
        out[self.standing] = 1.0
        return out

    def _phase(self):
        """Per-foot gait phase; standing envs pinned to the stance phase."""
        t = self.gait_phase
        ph = np.stack([t, (t + 0.5) % 1.0, (t + 0.5) % 1.0, t], axis=1)
        ph[self.standing] = STAND_PHASE
        return ph

    def obs(self, qpos=None, qvel=None):
        if qpos is None:
            qpos, qvel = self._gather()
        o = np.empty((self.n, NUM_OBS))
        dof_pos = np.concatenate([qpos[:, LEGS], qpos[:, ARM]], axis=1)
        default18 = np.concatenate([DEFAULT_ANGLES, ARM_DEFAULT])
        act18 = np.concatenate([self.actions, np.zeros((self.n, 6))], axis=1)
        roll, pitch = rpy_from_quat(qpos[:, 3:7])

        o[:, 0:18] = (dof_pos - default18) * DOF_POS_SCALE
        o[:, 18:30] = qvel[:, DOF_LEGS] * DOF_VEL_SCALE
        o[:, 30:48] = act18
        o[:, 48:51] = self.cmd * CMD_SCALE
        o[:, 51:57] = 0.0                     # arm command: zeros in stage 1
        o[:, 57] = roll
        o[:, 58] = pitch
        o[:, 59:63] = self.clock()
        return np.clip(o, -100.0, 100.0)

    def _push_hist(self, o):
        """Slide the 30-frame observation window by one control step."""
        self.obs_hist[:, :-NUM_OBS] = self.obs_hist[:, NUM_OBS:]
        self.obs_hist[:, -NUM_OBS:] = o

    def privileged(self):
        """Privileged observations for the asymmetric critic.

        Official uses priv_observe_friction + priv_observe_restitution,
        normalised as (x - shift) * scale with scale = 2/(max-min) and
        shift = (max+min)/2. Without domain randomisation these are constant.
        """
        return np.zeros((self.n, 2))

    # ---------------------------------------------------------------- step
    def step(self, actions):
        self.prev_actions = self.actions
        self.actions = np.clip(np.asarray(actions, dtype=np.float64), -10.0, 10.0)
        self.standing = np.all(np.abs(self.cmd) < ZERO_CMD_EPS, axis=1)

        off = ACTION_SCALE * self.actions
        off[:, HIP_IDX] *= HIP_SCALE
        targets = DEFAULT_ANGLES + off

        chunks = self._chunks()
        out = list(self._pool().map(self._sim_chunk, [(c, targets) for c in chunks]))
        energy_sum = np.empty((self.n, 12))
        foot_xyz = np.empty((self.n, 4, 3))
        foot_force = np.empty((self.n, 4))
        for (a, b), (en, fx, ff) in zip(chunks, out):
            energy_sum[a:b] = en
            foot_xyz[a:b] = fx
            foot_force[a:b] = ff

        # standing envs freeze their gait phase (paper SIII-B-1)
        advance = (self.gait_phase + CTRL_DT * self.gait_freq) % 1.0
        self.gait_phase = np.where(self.standing, self.gait_phase, advance)
        self.step_count += 1

        qpos, qvel = self._gather()
        rew, fail = self._reward(qpos, qvel, energy_sum / DECIM, foot_xyz, foot_force)
        done = fail | (self.step_count >= self.episode_steps)
        o = self.obs(qpos, qvel)
        self._push_hist(o)
        return (o, rew, done, dict(timeout=self.step_count >= self.episode_steps))

    def _sim_chunk(self, payload):
        span, targets = payload
        a, b = span
        en = np.empty((b - a, 12))
        fx = np.empty((b - a, 4, 3))
        ff = np.empty((b - a, 4))
        fw3 = np.zeros((b - a, 4, 3))      # world-frame force, for cross-checks
        fbody = self.foot_bodies
        fset = set(fbody)
        cfr = np.zeros(6)
        for k, i in enumerate(range(a, b)):
            d = self.data[i]
            m = self.models[i]
            tgt = targets[i]
            strength = self.motor_strength[i]
            offset = self.motor_offset[i]
            acc = np.zeros(12)
            for _ in range(DECIM):
                d.qpos[ARM] = ARM_DEFAULT
                d.qvel[DOF_ARM] = 0.0
                d.qpos[GRIP] = GRIP_DEFAULT
                d.qvel[24:26] = 0.0
                tau = np.clip((KP * (tgt + offset - d.qpos[LEGS])
                               - KD * d.qvel[DOF_LEGS]) * strength,
                              -TAU_MAX, TAU_MAX)
                d.ctrl[:12] = tau
                d.ctrl[12:18] = ARM_DEFAULT
                mujoco.mj_step(m, d)
                # paper Eq.(4) is sum |tau_i * qdot_i|^2: square per substep,
                # then average over the control step (not square of the mean)
                acc += (tau * d.qvel[DOF_LEGS]) ** 2
            en[k] = acc
            for j, f in enumerate(fbody):
                fx[k, j] = d.xpos[f]
                ff[k, j] = 0.0
            for c in range(d.ncon):
                b1 = m.geom_bodyid[d.contact[c].geom1]
                b2 = m.geom_bodyid[d.contact[c].geom2]
                j = -1
                if b1 in fset:
                    j = fbody.index(b1)
                elif b2 in fset:
                    j = fbody.index(b2)
                if j >= 0:
                    mujoco.mj_contactForce(m, d, c, cfr)
                    ff[k, j] += abs(cfr[0])          # normal force
                    # contact frame -> world, kept for the official-reward
                    # cross-check (rewards.py uses the 3D force norm)
                    # mj_contactForce gives the wrench on the first body; the
                    # sign is flipped so +z matches Isaac Gym's convention
                    # (force applied *to* the robot, positive upward)
                    fw3[k, j] -= d.contact[c].frame.reshape(3, 3) @ cfr[:3]
            self._fw3[a + k] = fw3[k]
        return en, fx, ff

    # -------------------------------------------------------------- reward
    def _reward(self, qpos, qvel, energy_rate, foot_xyz, foot_force):
        R = quat_to_matrix(qpos[:, 3:7])
        lin_vel = np.einsum("nji,nj->ni", R, qvel[:, 0:3])   # world -> base
        ang_vel = np.einsum("nji,nj->ni", R, qvel[:, 3:6])
        roll, pitch = rpy_from_quat(qpos[:, 3:7])
        self.base_lin_vel = lin_vel
        self.base_ang_vel = ang_vel

        # foot velocities by finite difference of body positions
        if self.prev_foot_xyz is None:
            foot_vel = np.zeros_like(foot_xyz)
        else:
            foot_vel = (foot_xyz - self.prev_foot_xyz) / CTRL_DT
        self.prev_foot_xyz = foot_xyz.copy()

        # gait phase per foot, order FR, FL, RR, RL  (FR/RL in phase).
        # Standing envs are pinned to STAND_PHASE so every foot is in stance
        # and the clearance target collapses to the foot radius.
        phase = self._phase()
        desired_contact = (phase < 0.5).astype(np.float64)     # WTW convention
        phases = np.abs(1.0 - 2.0 * phase) - 0.5               # in [-0.5, 0.5]

        cmd = self.cmd
        # every term is kept un-weighted in T so it can be compared one by one
        # against the official implementation (see crosscheck_rewards.py)
        T = {}
        T["track_lin"] = np.exp(-np.sum((lin_vel[:, :2] - cmd[:, :2]) ** 2, axis=1) / 0.25)
        T["track_ang"] = np.exp(-((ang_vel[:, 2] - cmd[:, 2]) ** 2) / 0.25)

        # ---- Raïbert foot-placement heuristic (rewards.py:255-288)
        dx = foot_xyz[:, :, 0] - qpos[:, 0:1]
        dy = foot_xyz[:, :, 1] - qpos[:, 1:2]
        yaw = np.arctan2(2 * (qpos[:, 3] * qpos[:, 6] + qpos[:, 4] * qpos[:, 5]),
                         1 - 2 * (qpos[:, 5] ** 2 + qpos[:, 6] ** 2))
        c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
        body_x = c * dx + s * dy
        body_y = -s * dx + c * dy

        stance_w, stance_l, freq = 0.3, 0.45, 3.0
        ys_nom = np.array([-0.15, 0.15, -0.15, 0.15])      # FR, FL, RR, RL
        xs_nom = np.array([0.225, 0.225, -0.225, -0.225])
        y_off = phases * (cmd[:, 2:3] * stance_l / 2) * (0.5 / freq)
        y_off[:, 2:] *= -1
        x_off = phases * cmd[:, 0:1] * (0.5 / freq)
        err = (xs_nom + x_off - body_x) ** 2 + (ys_nom + y_off - body_y) ** 2
        T["raibert"] = np.sum(err, axis=1)

        # ---- foot clearance (rewards.py:212-217).
        # The official function builds its own swing ramp, NOT the Raïbert
        # phases used above: 1 - |1 - clip(2f-1, 0, 1) * 2|, which is 0 during
        # stance and ramps 0->1 across swing.
        swing = 1.0 - np.abs(1.0 - np.clip(phase * 2.0 - 1.0, 0.0, 1.0) * 2.0)
        target_h = 0.04 * swing + 0.02
        T["clearance"] = np.sum((target_h - foot_xyz[:, :, 2]) ** 2
                                * (1 - desired_contact), axis=1)

        # ---- contact shaping (rewards.py:160-177)
        # rewards.py:160 uses the 3D force norm, not just the normal component
        force_norm = np.linalg.norm(self._fw3, axis=2)
        T["contact_force"] = np.mean((1 - desired_contact) *
                                     (1 - np.exp(-force_norm ** 2 / 100.0)), axis=1)
        T["contact_vel"] = np.mean(desired_contact *
                                   (1 - np.exp(-np.sum(foot_vel ** 2, axis=2) / 10.0)),
                                   axis=1)

        # ---- slip / smoothness / energy
        # rewards.py:192 measures the foot velocity while it is actually in
        # contact (normal force > 1 N), with a one-step latch, not while it is
        # *supposed* to be in contact
        in_contact = self._fw3[:, :, 2] > 1.0
        contact_filt = in_contact | self.last_contacts
        self.last_contacts = in_contact
        T["slip"] = np.sum(contact_filt * np.sum(foot_vel[:, :, :2] ** 2, axis=2), axis=1)
        T["lin_vel_z"] = lin_vel[:, 2] ** 2
        T["ang_vel_xy"] = np.sum(ang_vel[:, :2] ** 2, axis=1)
        T["orientation"] = roll ** 2 + pitch ** 2

        # rewards.py:240 -- track the commanded body attitude. The desired
        # quaternion is R_roll(-roll_cmd) * R_pitch(-pitch_cmd); we compare the
        # xy of the projected gravity, exactly as upstream does.
        qr = np.stack([np.cos(-self.body_cmd[:, 1] / 2),
                       np.sin(-self.body_cmd[:, 1] / 2),
                       np.zeros(self.n), np.zeros(self.n)], axis=1)
        qp = np.stack([np.cos(-self.body_cmd[:, 0] / 2),
                       np.zeros(self.n),
                       np.sin(-self.body_cmd[:, 0] / 2),
                       np.zeros(self.n)], axis=1)
        w1, x1, y1, z1 = qr[:, 0], qr[:, 1], qr[:, 2], qr[:, 3]
        w2, x2, y2, z2 = qp[:, 0], qp[:, 1], qp[:, 2], qp[:, 3]
        qd = np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                       w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                       w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                       w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], axis=1)
        Rd = quat_to_matrix(qd)
        # projected gravity = R^T @ [0,0,-1] = -(third row of R)
        pg = -R[:, 2, :]        # actual
        pg_d = -Rd[:, 2, :]     # desired
        T["orientation_control"] = np.sum((pg[:, :2] - pg_d[:, :2]) ** 2, axis=1)
        T["dof_vel"] = np.sum(qvel[:, DOF_LEGS] ** 2, axis=1)
        T["action_rate"] = np.sum((self.actions - self.prev_actions) ** 2, axis=1)
        # paper Eq.(4): energy_rate holds mean over substeps of (tau*qdot)^2
        T["energy"] = np.sum(energy_rate, axis=1)

        self.terms = T
        pos = 1.0 * T["track_lin"] + 0.5 * T["track_ang"]
        neg = (-10.0 * T["raibert"]
               - 30.0 * T["clearance"]
               - 4.0 * T["contact_force"]
               - 4.0 * T["contact_vel"]
               - 0.04 * T["slip"]
               - 0.02 * T["lin_vel_z"]
               - 0.001 * T["ang_vel_xy"]
               - 0.0 * T["orientation"]   # upstream wtw_config sets this to 0
               - 5.0 * T["orientation_control"]
               - 1e-4 * T["dof_vel"]
               - 0.01 * T["action_rate"]
               - 0.00004 * T["energy"])

        # official combination (legged_robot.py:578): rewards are NOT summed
        r = pos * np.exp(neg / SIGMA_REW_NEG)

        fail = (qpos[:, 2] < 0.18) | (np.abs(roll) > 1.0) | (np.abs(pitch) > 1.0)
        return r, fail

    def close(self):
        if self._thread_pool is not None:
            self._thread_pool.shutdown(wait=True)
            self._thread_pool = None
