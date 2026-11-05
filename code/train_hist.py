"""PPO with the official asymmetric actor-critic and 30-frame history.

Matches `go1_gym_learn/ppo_cse_automatic/dog_ac.py`:

    adaptation : 30*63 -> 256 -> 128 -> 2        (ELU)
    actor      : (30*63 + 2) -> 512 -> 256 -> 128 -> 12
    critic     : (30*63 + 2) -> 512 -> 256 -> 128 -> 1

The actor consumes `cat(history, adaptation(history))` (student path, what
`act()` / `update_distribution()` do at training time); the critic consumes
`cat(history, privileged)` (teacher path, `evaluate()`). That is the asymmetry
the paper describes.

Run:  python train_hist.py --envs 256 --iters 200 --dr
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

from model_io import load_model_path                      # noqa: E402
from envs.go1_env import Go1Env, NUM_OBS, NUM_ACT, HIST_LEN   # noqa: E402

RUNS = HERE / "runs"

CLIP = 0.2
ENTROPY = 0.01
EPOCHS = 5
MINIBATCHES = 4
GAMMA = 0.99
LAM = 0.95
STEPS_PER_ENV = 24
HIST_DIM = HIST_LEN * NUM_OBS          # 1890


class AsymActorCritic(nn.Module):
    def __init__(self, num_actions=NUM_ACT, priv_dim=2, hist_dim=HIST_DIM):
        super().__init__()
        self.adaptation = nn.Sequential(
            nn.Linear(hist_dim, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, priv_dim), nn.ELU(),
        )
        self.actor = self._mlp(hist_dim + priv_dim, 512, num_actions, 0.01)
        self.critic = self._mlp(hist_dim + priv_dim, 512, 1, 1.0)
        self.log_std = nn.Parameter(torch.zeros(num_actions))

    @staticmethod
    def _mlp(n_in, hidden, n_out, gain):
        layers, prev = [], n_in
        for h in (hidden, hidden // 2, hidden // 4):
            layers += [nn.Linear(prev, h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, n_out))
        net = nn.Sequential(*layers)
        nn.init.orthogonal_(net[-1].weight, gain=gain)
        nn.init.zeros_(net[-1].bias)
        return net

    def student_act(self, hist):
        mu = self.actor(torch.cat((hist, self.adaptation(hist)), dim=-1))
        return mu

    def value(self, hist, priv):
        return self.critic(torch.cat((hist, priv), dim=-1)).squeeze(-1)

    @torch.no_grad()
    def act(self, hist, priv):
        mu = self.student_act(hist)
        std = self.log_std.exp().expand_as(mu)
        dist = torch.distributions.Normal(mu, std)
        a = dist.sample()
        return a, dist.log_prob(a).sum(-1), self.value(hist, priv)

    @torch.no_grad()
    def mean_action(self, hist):
        return self.student_act(hist)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--torch-threads", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dr", action="store_true")
    ap.add_argument("--tag", default="hist")
    args = ap.parse_args()

    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = load_model_path(HERE / "models" / "arx5p2Go1.xml")
    if args.dr:
        from envs.go1_env_dr import Go1EnvDR
        env = Go1EnvDR(model, args.envs, num_threads=args.threads, seed=args.seed)
    else:
        env = Go1Env(model, args.envs, num_threads=args.threads, seed=args.seed)

    dev = torch.device("cpu")
    net = AsymActorCritic().to(dev)
    opt = optim.Adam(net.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"params={n_params:,}  envs={args.envs}  dr={args.dr}  lr={args.lr}")

    run_dir = RUNS / f"{args.tag}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(exist_ok=True)
    log_path = run_dir / "log.csv"
    with log_path.open("w", newline="") as fh:
        csv.writer(fh).writerow(["iter", "steps", "rew_mean", "vx_err",
                                 "ep_len", "policy_loss", "value_loss", "sps"])

    env.reset()
    hist = torch.as_tensor(env.obs_hist, dtype=torch.float32)
    priv = torch.as_tensor(env.privileged(), dtype=torch.float32)
    buf = {k: [] for k in ("hist", "priv", "act", "logp", "rew", "done", "val")}
    cur_ret = np.zeros(args.envs)
    cur_len = np.zeros(args.envs, dtype=np.int64)
    ep_rets, ep_lens = [], []
    total = 0
    t_start = time.time()

    for it in range(1, args.iters + 1):
        t0 = time.time()
        for k in buf:
            buf[k].clear()
        for _ in range(STEPS_PER_ENV):
            with torch.no_grad():
                a, logp, v = net.act(hist, priv)
            _, rew, done, _ = env.step(a.numpy())
            buf["hist"].append(hist)
            buf["priv"].append(priv)
            buf["act"].append(a)
            buf["logp"].append(logp)
            buf["rew"].append(torch.as_tensor(rew, dtype=torch.float32))
            buf["done"].append(torch.as_tensor(done.astype(np.float32)))
            buf["val"].append(v)
            cur_ret += rew
            cur_len += 1
            for i in np.where(done)[0]:
                ep_rets.append(float(cur_ret[i]))
                ep_lens.append(int(cur_len[i]))
                cur_ret[i] = 0.0
                cur_len[i] = 0
            if done.any():
                env.reset(np.where(done)[0])
            hist = torch.as_tensor(env.obs_hist, dtype=torch.float32)
            priv = torch.as_tensor(env.privileged(), dtype=torch.float32)

        total += STEPS_PER_ENV * args.envs
        with torch.no_grad():
            last_val = net.value(hist, priv)
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

        b_hist = torch.cat(buf["hist"])
        b_priv = torch.cat(buf["priv"])
        b_act = torch.cat(buf["act"])
        b_logp = torch.cat(buf["logp"])
        b_adv = adv.reshape(-1)
        b_ret = ret.reshape(-1)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        n = b_hist.shape[0]

        pl_sum = vl_sum = 0.0
        for _ in range(EPOCHS):
            perm = torch.randperm(n)
            for m in range(MINIBATCHES):
                idx = perm[m * n // MINIBATCHES:(m + 1) * n // MINIBATCHES]
                mu = net.student_act(b_hist[idx])
                std = net.log_std.exp().expand_as(mu)
                dist = torch.distributions.Normal(mu, std)
                logp = dist.log_prob(b_act[idx]).sum(-1)
                ent = dist.entropy().sum(-1).mean()
                v = net.value(b_hist[idx], b_priv[idx])
                ratio = (logp - b_logp[idx]).exp()
                s1 = ratio * b_adv[idx]
                s2 = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * b_adv[idx]
                pl = -torch.min(s1, s2).mean()
                vl = ((v - b_ret[idx]) ** 2).mean()
                loss = pl + 0.5 * vl - ENTROPY * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                pl_sum += float(pl)
                vl_sum += float(vl)

        sps = STEPS_PER_ENV * args.envs / (time.time() - t0)
        with torch.no_grad():
            mv = net.mean_action(hist).numpy()
        err = float(np.abs(env.base_lin_vel[:, 0] - env.cmd[:, 0]).mean())
        if it % 10 == 0 or it == 1:
            print(f"it {it:4d} | step {total/1e6:6.2f}M | rew {np.mean(ep_rets[-50:]):6.3f}"
                  f" | vx_err {err:5.3f} | ep_len {np.mean(ep_lens[-50:]):6.1f}"
                  f" | {sps:5.0f} sps")
        with log_path.open("a", newline="") as fh:
            csv.writer(fh).writerow([it, total, round(float(np.mean(ep_rets[-50:] or [0])), 4),
                                     round(err, 4), round(float(np.mean(ep_lens[-50:] or [0])), 1),
                                     round(pl_sum / (EPOCHS * MINIBATCHES), 4),
                                     round(vl_sum / (EPOCHS * MINIBATCHES), 4), int(sps)])
        if it % 25 == 0:
            torch.save({"model": net.state_dict(), "iter": it,
                        "hist": True}, run_dir / f"ckpt_{it}.pt")

    torch.save({"model": net.state_dict(), "iter": args.iters, "hist": True},
               run_dir / "final.pt")
    (run_dir / "summary.json").write_text(json.dumps(dict(
        params=n_params, envs=args.envs, iters=args.iters, dr=args.dr,
        lr=args.lr, elapsed_s=round(time.time() - t_start, 1)), indent=2))
    print("done ->", run_dir)


if __name__ == "__main__":
    main()
