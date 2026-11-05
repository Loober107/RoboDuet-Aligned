"""Evaluate a checkpoint produced by train_hist.py (history + asymmetric AC).

Same metrics as eval_stage1.py (steady-state window 4-6 s, duty factor, step
frequency) but it feeds the 1890-dim observation history and uses the student
actor path, which is what deployment uses.

Run:  python eval_hist.py --ckpt runs/hist_dr_.../ckpt_300.pt --vx 0.5
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from model_io import load_model_path                     # noqa: E402
from envs.go1_env import NUM_OBS, HIST_LEN               # noqa: E402
from train_hist import AsymActorCritic                   # noqa: E402

RES = HERE / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", default="hist")
    ap.add_argument("--vx", type=float, default=0.5)
    ap.add_argument("--envs", type=int, default=32)
    ap.add_argument("--steps", type=int, default=320)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--dr", action="store_true",
                    help="evaluate with domain randomisation on")
    ap.add_argument("--friction", type=float, default=None)
    ap.add_argument("--payload", type=float, default=None)
    args = ap.parse_args()

    model = load_model_path(HERE / "models" / "arx5p2Go1.xml")
    if args.dr or args.friction is not None or args.payload is not None:
        from envs.go1_env_dr import Go1EnvDR
        env = Go1EnvDR(model, args.envs, num_threads=4, seed=args.seed)
        env.freeze_motor()
        if args.friction is not None:
            env.set_friction(args.friction)
        if args.payload is not None:
            env.set_payload(args.payload)
    else:
        from envs.go1_env import Go1Env
        env = Go1Env(model, args.envs, num_threads=4, seed=args.seed)

    net = AsymActorCritic()
    ck = torch.load(HERE / args.ckpt, map_location="cpu", weights_only=False)
    net.load_state_dict(ck["model"])
    net.eval()

    env.reset()
    env.cmd[:] = [args.vx, 0.0, 0.0]

    vs, hs, foots = [], [], []
    for k in range(args.steps):
        hist = torch.as_tensor(env.obs_hist, dtype=torch.float32)
        with torch.no_grad():
            mu = net.mean_action(hist)
        env.step(mu.numpy())
        # paper SIV-E: ignore the first 4 s, average the next 2 s
        if 200 <= k < 300:
            vs.append(env.base_lin_vel.copy())
            hs.append(np.array([d.qpos[2] for d in env.data]))
            foots.append(env.prev_foot_xyz.copy())

    v = np.concatenate(vs)
    h = np.concatenate(hs)
    foot = np.concatenate(foots)

    # duty factor from foot height (same 0.025 m threshold as eval_stage1)
    contact = (foot[:, :, 2] < 0.025).astype(float)
    names = ["FR", "FL", "RR", "RL"]
    out = dict(ckpt=args.ckpt, iter=int(ck.get("iter", -1)),
               cmd=args.vx, dr=bool(args.dr),
               vx_mean=round(float(v[:, 0].mean()), 4),
               vy_mean=round(float(v[:, 1].mean()), 4),
               wz_mean=round(float(v[:, 2].mean()), 4),
               vx_err=round(float(np.abs(v[:, 0] - args.vx).mean()), 4),
               vy_err=round(float(np.abs(v[:, 1]).mean()), 4),
               wz_err=round(float(np.abs(v[:, 2]).mean()), 4),
               base_z_mean=round(float(h.mean()), 4),
               foot_z_max=round(float(foot[:, :, 2].max()), 4),
               duty={}, step_freq_hz={})
    for i, n in enumerate(names):
        c = contact[:, i]
        out["duty"][n] = round(float(c.mean()), 3)
        tr = np.diff((c > 0.5).astype(int))
        lifts = (tr == -1).sum()
        dur = len(tr) * 0.02
        out["step_freq_hz"][n] = round(float(lifts / dur), 2) if dur > 0 else 0.0

    print(json.dumps(out, indent=2, ensure_ascii=False))
    (RES / f"eval_{args.tag}.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    env.close()


if __name__ == "__main__":
    main()
