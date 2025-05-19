import argparse, os, random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dmc import make_dmc_env
from tqdm import tqdm


# ────────────────────────────────────────────────────────────────────────────────
# Hyper-parameters  (tweak as desired)
# ────────────────────────────────────────────────────────────────────────────────
HIDDEN1, HIDDEN2 = 400, 300       
ACTOR_LR = 1e-4
CRITIC_LR = 1e-3
GAMMA = 0.99
TAU = 5e-3                      
REPLAY_SIZE = int(1e6)
BATCH_SIZE = 512                 
START_STEPS = 20_000                 # purely random before training
MAX_STEPS = 1_000_000
EVAL_INTERVAL = 50_000
NOISE_STD_INIT = 0.2                 # exploration noise σ
NOISE_STD_FINAL = 0.05
SCHED_GAMMA = 0.9999                 # LR exponential decay

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple("Transition",
                        "state action reward next_state done")


# ────────────────────────────────────────────────────────────────────────────────
# Replay buffer
# ────────────────────────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def push(self, *args):
        self.buf.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        t = Transition(*zip(*batch))
        s      = torch.as_tensor(np.stack(t.state),      dtype=torch.float32, device=device)
        a      = torch.as_tensor(np.stack(t.action),     dtype=torch.float32, device=device)
        r      = torch.as_tensor(np.array(t.reward),     dtype=torch.float32, device=device).unsqueeze(1)
        s2     = torch.as_tensor(np.stack(t.next_state), dtype=torch.float32, device=device)
        d      = torch.as_tensor(np.array(t.done),       dtype=torch.float32, device=device).unsqueeze(1)
        return s, a, r, s2, d

    def __len__(self):
        return len(self.buf)


# ────────────────────────────────────────────────────────────────────────────────
# Networks
# ────────────────────────────────────────────────────────────────────────────────
def mlp(in_dim, out_dim, hidden1=HIDDEN1, hidden2=HIDDEN2, final_tanh=False):
    layers = [ nn.Linear(in_dim, hidden1), nn.ReLU(),
               nn.Linear(hidden1, hidden2), nn.ReLU(),
               nn.Linear(hidden2, out_dim) ]
    if final_tanh: layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_max):
        super().__init__()
        self.net = mlp(state_dim, action_dim, final_tanh=True)
        self.scale = action_max

    def forward(self, s):
        return self.net(s) * self.scale


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = mlp(state_dim + action_dim, 1)

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))


# ────────────────────────────────────────────────────────────────────────────────
# Utils
# ────────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def soft_update(target, src, tau):
    for tg, src_p in zip(target.parameters(), src.parameters()):
        tg.data.mul_(1 - tau).add_(src_p.data, alpha=tau)


@torch.no_grad()
def evaluate(actor, episodes=20):
    env = make_env()
    returns = []
    for _ in range(episodes):
        s, _ = env.reset(seed=np.random.randint(0, 1_000_000))
        ep_r, done = 0.0, False
        while not done:
            a = actor(torch.as_tensor(s, dtype=torch.float32,
                                      device=device).unsqueeze(0))
            s, r, term, trunc, _ = env.step(a.cpu().numpy()[0])
            ep_r += r
            done = term or trunc
        returns.append(ep_r)
    env.close()
    r = np.array(returns)
    return float(r.mean() - r.std())


def make_env(seed=None):
    env = make_dmc_env("humanoid-walk",
                       seed if seed is not None else np.random.randint(0, 1_000_000),
                       flatten=True, use_pixels=False)
    return env


# ────────────────────────────────────────────────────────────────────────────────
# Training loop
# ────────────────────────────────────────────────────────────────────────────────
def train(seed=0):
    random.seed(seed);  np.random.seed(seed);  torch.manual_seed(seed)

    env = make_env(seed)
    state_dim  = env.observation_space.shape[0]      # 67
    action_dim = env.action_space.shape[0]           # 21
    action_max = float(env.action_space.high[0])     # 1.0

    # networks + target copies
    actor  = Actor(state_dim, action_dim, action_max).to(device)
    critic = Critic(state_dim, action_dim).to(device)
    act_t  = Actor(state_dim, action_dim, action_max).to(device)
    crt_t  = Critic(state_dim, action_dim).to(device)
    act_t.load_state_dict(actor.state_dict())
    crt_t.load_state_dict(critic.state_dict())

    act_optim = optim.Adam(actor.parameters(),  lr=ACTOR_LR)
    crt_optim = optim.Adam(critic.parameters(), lr=CRITIC_LR)

    # exponential LR schedulers
    act_sched = optim.lr_scheduler.ExponentialLR(act_optim, gamma=SCHED_GAMMA)
    crt_sched = optim.lr_scheduler.ExponentialLR(crt_optim, gamma=SCHED_GAMMA)

    buf = ReplayBuffer(REPLAY_SIZE)

    s, _ = env.reset()
    ep_r, ep_len, ep = 0.0, 0, 0
    best_score = -float("inf")
    pbar = tqdm(range(1, MAX_STEPS + 1))

    for t in pbar:
        # exploration noise annealed linearly
        frac = min(1.0, t / MAX_STEPS)
        sigma = NOISE_STD_INIT + (NOISE_STD_FINAL - NOISE_STD_INIT) * frac
        if t < START_STEPS:
            a = env.action_space.sample()
        else:
            with torch.no_grad():
                a = actor(torch.as_tensor(s, dtype=torch.float32,
                                          device=device).unsqueeze(0))
                a = a.cpu().numpy()[0]
            a = (a + np.random.normal(0, sigma, size=action_dim)
                 ).clip(-action_max, action_max)

        s2, r, term, trunc, _ = env.step(a)
        done = term or trunc
        buf.push(s, a, r, s2, float(done))

        s = s2
        ep_r += r;  ep_len += 1

        # episode end
        if done:
            pbar.set_description(f"Ep {ep:4d} | len {ep_len:4d} | return {ep_r:9.1f}")
            s, _ = env.reset()
            ep_r, ep_len, ep = 0.0, 0, ep + 1

        # update after warm-up
        if t >= START_STEPS and len(buf) >= BATCH_SIZE:
            # sample
            ss, aa, rr, ss2, dd = buf.sample(BATCH_SIZE)

            # target actions (no policy noise in plain DDPG target)
            with torch.no_grad():
                a2 = act_t(ss2)
                q2 = crt_t(ss2, a2)
                y = rr + GAMMA * (1 - dd) * q2

            # critic update
            q = critic(ss, aa)
            crt_loss = nn.functional.mse_loss(q, y)
            crt_optim.zero_grad()
            crt_loss.backward()
            crt_optim.step()

            # actor update (policy gradient)
            act_loss = -critic(ss, actor(ss)).mean()
            act_optim.zero_grad()
            act_loss.backward()
            act_optim.step()

            # target net slow update
            soft_update(act_t, actor, TAU)
            soft_update(crt_t, critic, TAU)

            # LR decay
            act_sched.step()
            crt_sched.step()

        # quick deterministic eval
        if t % EVAL_INTERVAL == 0:
            score = evaluate(actor, episodes=10)
            print(f"[eval {t:7d}] mean-std score: {score:8.1f}")
            if score > best_score:
                best_score = score
                torch.save(actor.state_dict(), "ddpg_humanoid_actor.pth")
                print("  ↳ new best saved  (score ↑)")
    env.close()
    print(f"Training finished. Best score = {best_score:.1f}")


# ────────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train(seed=args.seed)

