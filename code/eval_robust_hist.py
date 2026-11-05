"""Robustness sweep for the history/asymmetric model (train_hist checkpoints).

Same grid and metric as eval_robust.py, but feeds the 1890-dim observation
history and uses the student actor path.

Run:  python eval_robust_hist.py --ckpt runs/hist_dr_.../ckpt_600.pt --tag hist600
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from model_io import load_model_path                        # noqa: E402
from envs.go1_env_dr import Go1EnvDR                        # noqa: E402
from train_hist import AsymActorCritic                      # noqa: E402


def evaluate(model, ckpt, net, n_env, seed, vx, friction, payload, steps=320):
    env = Go1EnvDR(model, n_env, num_threads=4, seed=seed)
    env.freeze_motor()
    env.set_restitution(0.0)
    env.set_friction(friction)
    env.set_payload(payload)
    env.grav_impulse = np.zeros(3)
    env._apply_gravity()
    env.reset()
    env.cmd[:] = [vx, 0.0, 0.0]

    vs, hs, fallen = [], [], 0
    for k in range(steps):
        hist = torch.as_tensor(env.obs_hist, dtype=torch.float32)
        with torch.no_grad():
            mu = net.mean_action(hist)
        _, _, done, _ = env.step(mu.numpy())
        fallen += int(np.sum(done))
        if 200 <= k < 300:
            vs.append(env.base_lin_vel.copy())
            hs.append(np.array([d.qpos[2] for d in env.data]))
    env.close()
    v = np.concatenate(vs)
    h = np.concatenate(hs)
    return dict(vx_mean=float(v[:, 0].mean()),
                vx_err=float(np.abs(v[:, 0] - vx).mean()),
                h_mean=float(h.mean()), fallen=fallen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--vx", type=float, default=0.5)
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=21)
    args = ap.parse_args()

    model = load_model_path(HERE / "models" / "arx5p2Go1.xml")
    net = AsymActorCritic()
    ck = torch.load(HERE / args.ckpt, map_location="cpu", weights_only=False)
    net.load_state_dict(ck["model"])
    net.eval()

    grid = [(0.2, 0.0), (0.5, 0.0), (1.0, 0.0), (2.0, 0.0), (2.8, 0.0),
            (1.0, -2.0), (1.0, 2.0)]
    out = dict(ckpt=args.ckpt, cmd=args.vx, conditions=[])
    print(f"{'摩擦':>6}{'载荷':>7}{'实测vx':>10}{'误差':>9}{'高度':>9}{'摔倒':>7}")
    for fr, pl in grid:
        r = evaluate(model, args.ckpt, net, args.envs, args.seed, args.vx, fr, pl)
        r.update(friction=fr, payload=pl)
        out["conditions"].append(r)
        print(f"{fr:>6.2f}{pl:>+7.1f}{r['vx_mean']:>10.4f}{r['vx_err']:>9.4f}"
              f"{r['h_mean']:>9.4f}{r['fallen']:>7d}")
    errs = [c["vx_err"] for c in out["conditions"]]
    out["mean_err"] = float(np.mean(errs))
    out["worst_err"] = float(np.max(errs))
    print(f"\n平均误差 {out['mean_err']:.4f}   最差 {out['worst_err']:.4f}")
    p = HERE / "results" / f"robust_{args.tag}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", p)


if __name__ == "__main__":
    main()
