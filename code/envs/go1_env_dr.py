"""Domain randomisation for the Stage-1 env, matching the official config.

Values are the ones that survive the config chain
`config_go1 -> config_wtw -> config_asset -> scripts/auto_train.py`:

    friction            U[0.1, 3.0]        sampled once per env, fixed
    restitution         U[0.0, 0.4]        sampled once per env, fixed
    base payload        U[-2.0, 2.0] kg    sampled once per env, fixed
    motor strength      U[0.9, 1.1]        resampled every 4 s
    motor offset        U[-0.02, 0.02] rad resampled every 4 s
    gravity impulse     3D U[-1, 1] added to -9.8, applied 0.99 s every 8 s
    push robots         disabled           (wtw_config.py:53)
    lag timesteps       disabled           (auto_train.py:59)
    Kp/Kd factors       disabled
    com displacement    disabled

Re-randomising the rigid props *after* the start is disabled upstream
(`wtw_config.py:21 randomize_rigids_after_start = False`), so friction,
restitution and payload stay constant for an entire run — we replicate that.

Implementation note: MuJoCo physics parameters live on MjModel, which we
otherwise share across all envs. Here each env gets its own MjModel copy, but
the large constant arrays (meshes, BVHs, body/joint frames — 8.5 MB of the
8.54 MB total) are re-bound to the shared arrays, so a copy costs ~40 kB.
"""
import copy

import numpy as np

from envs.go1_env import Go1Env, LEGS, DOF_LEGS

# arrays that never change between envs and can be safely shared
SHARE_PREFIX = (
    "mesh_", "bvh_", "tex_", "geom_rbound", "geom_pos", "geom_quat", "geom_size",
    "geom_aabb", "geom_dataid", "geom_bodyid", "geom_type", "body_pos", "body_quat",
    "body_ipos", "body_iquat", "body_parentid", "body_jntnum", "body_jntadr",
    "jnt_pos", "jnt_axis", "jnt_range", "jnt_type", "jnt_bodyid", "dof_pos",
    "dof_axis", "dof_jntid", "dof_bodyid", "dof_type", "site_pos", "site_quat",
    "site_bodyid", "cam_", "light_", "name_", "actuator_", "mat_", "pair_",
)

# official values
FRICTION_RANGE = (0.1, 3.0)
RESTITUTION_RANGE = (0.0, 0.4)
PAYLOAD_RANGE = (-2.0, 2.0)
MOTOR_STRENGTH_RANGE = (0.9, 1.1)
MOTOR_OFFSET_RANGE = (-0.02, 0.02)
GRAVITY_RANGE = (-1.0, 1.0)
GRAVITY_INTERVAL_S = 8.0
GRAVITY_DURATION_S = 0.99
RAND_INTERVAL_S = 4.0
TRUNK_BODY_NAME = "trunk"


class Go1EnvDR(Go1Env):
    """Go1Env with the official domain randomisation switched on."""

    def __init__(self, model, num_envs, num_threads=8, seed=0, episode_steps=None,
                 enable=True):
        super().__init__(model, num_envs, num_threads, seed, episode_steps)
        self.dr_enabled = enable
        if not enable:
            return

        rng = np.random.default_rng(seed + 9973)
        self.domain = dict(
            friction=rng.uniform(*FRICTION_RANGE, size=self.n),
            restitution=rng.uniform(*RESTITUTION_RANGE, size=self.n),
            payload=rng.uniform(*PAYLOAD_RANGE, size=self.n),
        )
        # gravity impulse state (global, as upstream: one sim parameter)
        self.grav_impulse = np.zeros(3)
        self._grav = np.tile(np.array([0.0, 0.0, -9.81]), (self.n, 1))

        trunk = self._body_id(TRUNK_BODY_NAME)
        base_mass = model.body_mass[trunk]
        self.trunk_id = trunk
        self.base_mass = base_mass
        models = []
        for i in range(self.n):
            m = copy.deepcopy(model)
            for name in SHARE_PREFIX:
                v = getattr(model, name, None)
                if isinstance(v, np.ndarray):
                    try:
                        setattr(m, name, v)
                    except Exception:
                        pass
            # friction + restitution on every robot geom (floor is body 0)
            # MuJoCo has no restitution parameter; bounce is expressed through
            # the constraint damping ratio in geom_solref[:, 1] (1 = critically
            # damped, i.e. no bounce). Map restitution [0, 0.4] -> [1.0, 0.2].
            dampratio = float(np.clip(1.0 - 2.0 * self.domain["restitution"][i], 0.1, 1.0))
            # MuJoCo combines the two contact surfaces with an element-wise
            # MAX, so randomising only the robot geoms has no effect while the
            # floor keeps its default. Set every geom (floor included) so the
            # sampled value becomes the effective contact friction.
            for g in range(m.ngeom):
                m.geom_friction[g, 0] = self.domain["friction"][i]
                if m.geom_bodyid[g] != 0:
                    m.geom_solref[g, 1] = dampratio
            m.body_mass[trunk] = base_mass + self.domain["payload"][i]
            models.append(m)
        self.models = models
        import mujoco
        self.data = [mujoco.MjData(m) for m in models]
        # body_mass feeds cached constants (body_subtreemass, invweight0); they
        # have to be recomputed after the payload is applied
        for m, d in zip(models, self.data):
            mujoco.mj_setConst(m, d)
        for d in self.data:
            mujoco.mj_resetDataKeyframe(models[0], d, 0)
        self._apply_gravity()

        # motor params are resampled every 4 s (upstream `_randomize_dof_props`)
        self.motor_strength = rng.uniform(*MOTOR_STRENGTH_RANGE, size=(self.n, 12))
        self.motor_offset = rng.uniform(*MOTOR_OFFSET_RANGE, size=(self.n, 12))
        self._rand_every = int(round(RAND_INTERVAL_S / 0.02))
        self._grav_every = int(round(GRAVITY_INTERVAL_S / 0.02))
        self._grav_len = int(round(GRAVITY_DURATION_S / 0.02))

    # ------------------------------------------------------------------ util
    def _body_id(self, name):
        import mujoco
        for i in range(self.m.nbody):
            if mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, i) == name:
                return i
        raise ValueError(f"body {name!r} not found")

    def set_payload(self, values):
        """Force every env to a given added mass (kg) and refresh the cache."""
        import mujoco
        values = np.broadcast_to(np.asarray(values, dtype=float), (self.n,))
        self.domain["payload"] = values.copy()
        for i, (m, d) in enumerate(zip(self.models, self.data)):
            m.body_mass[self.trunk_id] = self.base_mass + values[i]
            mujoco.mj_setConst(m, d)

    def set_friction(self, values):
        values = np.broadcast_to(np.asarray(values, dtype=float), (self.n,))
        self.domain["friction"] = values.copy()
        for i, m in enumerate(self.models):
            for g in range(m.ngeom):
                m.geom_friction[g, 0] = values[i]

    def set_restitution(self, values):
        values = np.broadcast_to(np.asarray(values, dtype=float), (self.n,))
        self.domain["restitution"] = values.copy()
        for i, m in enumerate(self.models):
            damp = float(np.clip(1.0 - 2.0 * values[i], 0.1, 1.0))
            for g in range(m.ngeom):
                if m.geom_bodyid[g] != 0:
                    m.geom_solref[g, 1] = damp

    def set_motor(self, strength, offset=0.0):
        self.motor_strength[:] = strength
        self.motor_offset[:] = offset

    def privileged(self):
        """Privileged obs for the asymmetric critic.

        Matches priv_observe_friction + priv_observe_restitution, normalised as
        (x - shift) * scale with scale = 2/(max-min), shift = (max+min)/2.
        """
        f = (self.domain["friction"] - 0.5) * 2.0          # range [0, 1]
        r = (self.domain["restitution"] - 0.2) * 5.0       # range [0, 0.4]
        return np.stack([f, r], axis=1)

    def _apply_gravity(self):
        g = np.array([0.0, 0.0, -9.81]) + self.grav_impulse
        for m in self.models:
            m.opt.gravity[:] = g

    def _resample_motor(self, idx=None):
        rng = self.rng
        idx = np.arange(self.n) if idx is None else np.atleast_1d(idx)
        k = len(idx)
        self.motor_strength[idx] = rng.uniform(*MOTOR_STRENGTH_RANGE, size=(k, 12))
        self.motor_offset[idx] = rng.uniform(*MOTOR_OFFSET_RANGE, size=(k, 12))

    # ------------------------------------------------------------------ step
    def step(self, actions):
        t = int(self.step_count.max()) if self.n else 0
        if self.dr_enabled:
            # global gravity impulse: on every 8 s, cleared after 0.99 s
            if t % self._grav_every == 0:
                self.grav_impulse = self.rng.uniform(*GRAVITY_RANGE, size=3)
                self._apply_gravity()
            elif t % self._grav_every == self._grav_len:
                self.grav_impulse = np.zeros(3)
                self._apply_gravity()
            # per-env motor params every 4 s
            if t % self._rand_every == 0:
                self._resample_motor()
        return super().step(actions)

    # ----------------------------------------------------------------- reset
    def reset(self, idx=None):
        out = super().reset(idx)
        if self.dr_enabled:
            self._resample_motor(idx)
        return out

    def freeze_motor(self):
        """Stop resampling motor params (used by the verification script)."""
        self._resample_motor = lambda idx=None: None
