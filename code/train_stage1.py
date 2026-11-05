"""Stage-1 training (locomotion only) for the RoboDuet reproduction.

PPO hyperparameters and the rollout length are taken from the official code
(go1_gym_learn/ppo_cse_automatic/ppo.py:16-25 and __init__.py:49).

Run:  python train_stage1.py --envs 1024 --iters 3000
"""
import argparse
import csv
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from envs.go1_env import Go1Env, NUM_OBS, NUM_ACT, CMD_SCALE  # noqa: E402
from model_io import load_model_path  # noqa: E402

RUNS = HERE / "runs"
RUNS.mkdir(exist_ok=True)

# ---- official PPO settings
CLIP = 0.2
ENTROPY = 0.01
EPOCHS = 5
MINIBATCHES = 4
LR = 5e-4
GAMMA = 0.99
LAM = 0.95
STEPS_PER_ENV = 24
HIDDEN = (512, 256, 128)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=HIDDEN):
        super().__init__()
        def mlp(out, last_gain=1.0):
            layers, prev = [], obs_dim
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.ELU()]
                prev = h
            layers.append(nn.Linear(prev, out))
            net = nn.Sequential(*layers)
            nn.init.orthogonal_(net[-1].weight, gain=last_gain)
            nn.init.zeros_(net[-1].bias)
            return net
        self.actor = mlp(act_dim, 0.01)
        self.critic = mlp(1, 1.0)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):
        return self.actor(obs), self.critic(obs).squeeze(-1)

    @torch.no_grad()
    def act(self, obs):
        mu, v = self(obs)
        std = self.log_std.exp().expand_as(mu)
        dist = torch.distributions.Normal(mu, std)
        a = dist.sample()
        return a, dist.log_prob(a).sum(-1), v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--torch-threads", type=int, default=4)
    ap.add_argument("--tag", default="stage1")
    ap.add_argument("--resume", default="", help="checkpoint to warm-start from")
    ap.add_argument("--dr", action="store_true", help="enable domain randomisation")
    ap.add_argument("--lr", type=float, default=LR)
    args = ap.parse_args()

    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = load_model_path(HERE / "models" / "arx5p2Go1.xml")
    if args.dr:
        from envs.go1_env_dr import Go1EnvDR
        env = Go1EnvDR(model, args.envs, num_threads=args.threads, seed=args.seed)
        print("domain randomisation: ON")
    else:
        env = Go1Env(model, args.envs, num_threads=args.threads, seed=args.seed)

    dev = torch.device(args.device)
    net = ActorCritic(NUM_OBS, NUM_ACT).to(dev)
    if args.resume:
        ck = torch.load(HERE / args.resume, map_location="cpu", weights_only=False)
        net.load_state_dict(ck["model"])
        print(f"warm start from {args.resume} (iter {ck.get('iter', '?')})")
    opt = optim.Adam(net.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in net.parameters())

    obs = torch.as_tensor(env.reset(), dtype=torch.float32, device=dev)
    buf = {k: [] for k in ("obs", "act", "logp", "rew", "done", "val")}

    run_dir = RUNS / f"{args.tag}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(exist_ok=True)
    log_path = run_dir / "log.csv"
    with log_path.open("w", newline="") as fh:
        csv.writer(fh).writerow(
            ["iter", "steps", "rew_mean", "ret_mean", "ep_len",
             "vx_err", "vy_err", "wz_err", "policy_loss", "value_loss",
             "entropy", "sps", "elapsed_s"])

    print(f"n_params={n_params:,}  envs={args.envs}  threads={args.threads}")
    total_steps = 0
    ep_returns, ep_lens = [], []
    cur_ret = np.zeros(args.envs)
    cur_len = np.zeros(args.envs, dtype=np.int64)
    t_start = time.time()
    best_ret = -1e9

    for it in range(1, args.iters + 1):
        t0 = time.time()
        for k in buf:
            buf[k].clear()

        for _ in range(STEPS_PER_ENV):
            with torch.no_grad():
                act, logp, val = net.act(obs)
            a_np = act.cpu().numpy()
            o_np, rew, done, _ = env.step(a_np)

            buf["obs"].append(obs)
            buf["act"].append(act)
            buf["logp"].append(logp)
            buf["rew"].append(torch.as_tensor(rew, dtype=torch.float32, device=dev))
            buf["done"].append(torch.as_tensor(done.astype(np.float32), device=dev))
            buf["val"].append(val)

            cur_ret += rew
            cur_len += 1
            for i in np.where(done)[0]:
                ep_returns.append(float(cur_ret[i]))
                ep_lens.append(int(cur_len[i]))
                cur_ret[i] = 0.0
                cur_len[i] = 0
            if done.any():
                o_np = env.reset(np.where(done)[0])
            obs = torch.as_tensor(o_np, dtype=torch.float32, device=dev)

        total_steps += STEPS_PER_ENV * args.envs
        with torch.no_grad():
            _, last_val = net(obs)

        # ---- GAE
        vals = torch.stack(buf["val"])
        rews = torch.stack(buf["rew"])
        dones = torch.stack(buf["done"])
        adv = torch.zeros_like(rews)
        last = torch.zeros_like(last_val)
        for t in reversed(range(STEPS_PER_ENV)):
            nxt = last_val if t == STEPS_PER_ENV - 1 else vals[t + 1]
            nonterm = 1.0 - dones[t]
            delta = rews[t] + GAMMA * nxt * nonterm - vals[t]
            last = delta + GAMMA * LAM * nonterm * last
            adv[t] = last
        ret = adv + vals

        b_obs = torch.cat(buf["obs"])
        b_act = torch.cat(buf["act"])
        b_logp = torch.cat(buf["logp"])
        b_adv = adv.reshape(-1)
        b_ret = ret.reshape(-1)

        n = b_obs.shape[0]
        mb = n // MINIBATCHES
        pl = vl = ent = 0.0
        for _ in range(EPOCHS):
            perm = torch.randperm(n, device=dev)
            for m in range(MINIBATCHES):
                idx = perm[m * mb:(m + 1) * mb]
                mu, v = net(b_obs[idx])
                std = net.log_std.exp().expand_as(mu)
                dist = torch.distributions.Normal(mu, std)
                logp = dist.log_prob(b_act[idx]).sum(-1)
                ratio = (logp - b_logp[idx]).exp()
                a = b_adv[idx]
                a = (a - a.mean()) / (a.std() + 1e-8)
                s1 = ratio * a
                s2 = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * a
                pl = -torch.min(s1, s2).mean()
                vl = ((v - b_ret[idx]) ** 2).mean()
                ent = dist.entropy().sum(-1).mean()
                loss = pl + 0.5 * vl - ENTROPY * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()

        dt = time.time() - t0
        sps = STEPS_PER_ENV * args.envs / dt
        rew_mean = float(rews.mean())

        if it % 10 == 0 or it == 1:
            vx = env.base_lin_vel[:, 0]
            vy = env.base_lin_vel[:, 1]
            wz = env.base_ang_vel[:, 2]
            vx_err = float(np.mean(np.abs(vx - env.cmd[:, 0])))
            vy_err = float(np.mean(np.abs(vy - env.cmd[:, 1])))
            wz_err = float(np.mean(np.abs(wz - env.cmd[:, 2])))
            ret_mean = float(np.mean(ep_returns[-50:])) if ep_returns else 0.0
            len_mean = float(np.mean(ep_lens[-50:])) if ep_lens else 0.0
            with log_path.open("a", newline="") as fh:
                csv.writer(fh).writerow(
                    [it, total_steps, round(rew_mean, 4), round(ret_mean, 3),
                     round(len_mean, 1), round(vx_err, 3), round(vy_err, 3),
                     round(wz_err, 3), round(float(pl.detach()), 4), round(float(vl.detach()), 4),
                     round(float(ent), 3), int(sps), round(time.time() - t_start, 1)])
            print(f"it {it:5d} | step {total_steps/1e6:6.2f}M | rew {rew_mean:7.3f} "
                  f"| vx_err {vx_err:.2f} vy_err {vy_err:.2f} wz_err {wz_err:.2f} "
                  f"| ep_len {len_mean:5.1f} | {sps:6.0f} sps", flush=True)

        if ret_mean > best_ret and it % 10 == 0:
            best_ret = ret_mean
            torch.save({"model": net.state_dict(), "iter": it,
                        "args": vars(args)}, run_dir / "best.pt")
        if it % 100 == 0:
            torch.save({"model": net.state_dict(), "iter": it,
                        "args": vars(args)}, run_dir / f"ckpt_{it}.pt")

    torch.save({"model": net.state_dict(), "iter": args.iters,
                "args": vars(args)}, run_dir / "final.pt")
    (run_dir / "summary.json").write_text(json.dumps(dict(
        envs=args.envs, iters=args.iters, total_steps=total_steps,
        n_params=n_params, wall_s=round(time.time() - t_start, 1),
        last_vx_err=vx_err, last_vy_err=vy_err, last_wz_err=wz_err,
        ep_return_mean=float(np.mean(ep_returns[-50:])) if ep_returns else 0.0,
    ), indent=2))
    env.close()
    print(f"done -> {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

