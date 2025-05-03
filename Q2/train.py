import argparse
import os
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dmc import make_dmc_env
from tqdm import tqdm


# ────────────────────────────────────────────────────────────────────────────────
# Hyper-parameters 
# ────────────────────────────────────────────────────────────────────────────────
HIDDEN_SIZE   = 256
ACTOR_LR      = 3e-4
CRITIC_LR     = 3e-4
GAMMA         = 0.99
TAU           = 0.005
POLICY_DELAY  = 2
POLICY_NOISE  = 0.2
NOISE_CLIP    = 0.5
REPLAY_SIZE   = int(1e6)
BATCH_SIZE    = 256
START_STEPS   = 10_000          # purely random actions before TD3 updates
MAX_STEPS     = 500_000
EVAL_INTERVAL = 20_000

device = torch.device("cuda" if torch.cuda.is_available() else exit(1))
Transition = namedtuple("Transition",
                        "state action reward next_state done")


# ────────────────────────────────────────────────────────────────────────────────
# Experience buffer
# ────────────────────────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def push(self, *args):
        self.buf.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        t = Transition(*zip(*batch))

        states      = torch.as_tensor(np.stack(t.state),      dtype=torch.float32, device=device)
        actions     = torch.as_tensor(np.stack(t.action),     dtype=torch.float32, device=device)
        rewards     = torch.as_tensor(np.array(t.reward),     dtype=torch.float32, device=device).unsqueeze(1)
        next_states = torch.as_tensor(np.stack(t.next_state), dtype=torch.float32, device=device)
        dones       = torch.as_tensor(np.array(t.done),       dtype=torch.float32, device=device).unsqueeze(1)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buf)


# ────────────────────────────────────────────────────────────────────────────────
# Networks
# ────────────────────────────────────────────────────────────────────────────────
def mlp(in_dim, out_dim, hidden=HIDDEN_SIZE, final_tanh=False):
    layers = [
        nn.Linear(in_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, out_dim)
    ]
    if final_tanh:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_max):
        super().__init__()
        self.net = mlp(state_dim, action_dim, final_tanh=True)
        self.action_max = action_max

    def forward(self, s):
        return self.net(s) * self.action_max


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = mlp(state_dim + action_dim, 1)

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))


# ────────────────────────────────────────────────────────────────────────────────
# TD3 helpers
# ────────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def soft_update(target, src, tau):
    for tp, p in zip(target.parameters(), src.parameters()):
        tp.data.mul_(1 - tau).add_(p.data, alpha=tau)


def td3_update(buf, actor, act_t, c1, c2, c1_t, c2_t,
               act_opt, c1_opt, c2_opt, step):
    if len(buf) < BATCH_SIZE:
        return

    s, a, r, s2, d = buf.sample(BATCH_SIZE)

    # target policy smoothing
    noise = (torch.randn_like(a) * POLICY_NOISE).clamp(-NOISE_CLIP, NOISE_CLIP)
    a2 = (act_t(s2) + noise).clamp(-1.0, 1.0)

    with torch.no_grad():
        q1_t = c1_t(s2, a2)
        q2_t = c2_t(s2, a2)
        q_t  = torch.min(q1_t, q2_t)
        y = r + GAMMA * (1 - d) * q_t

    q1 = c1(s, a)
    q2 = c2(s, a)
    critic_loss = nn.functional.mse_loss(q1, y) + nn.functional.mse_loss(q2, y)

    c1_opt.zero_grad()
    c2_opt.zero_grad()
    critic_loss.backward()
    c1_opt.step()
    c2_opt.step()

    # delayed actor update
    if step % POLICY_DELAY == 0:
        act_loss = -c1(s, actor(s)).mean()
        act_opt.zero_grad()
        act_loss.backward()
        act_opt.step()

        soft_update(act_t, actor, TAU)
        soft_update(c1_t, c1, TAU)
        soft_update(c2_t, c2, TAU)


# ────────────────────────────────────────────────────────────────────────────────
# Environment factory (state observations, flattened)
# ────────────────────────────────────────────────────────────────────────────────
def make_env(seed=None):
    env_name = "cartpole-balance"
    env = make_dmc_env(env_name,
                       seed if seed is not None else np.random.randint(0, 1_000_000),
                       flatten=True,
                       use_pixels=False)
    return env


# ────────────────────────────────────────────────────────────────────────────────
# Training loop
# ────────────────────────────────────────────────────────────────────────────────
def train(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = make_env(seed)
    state_dim  = env.observation_space.shape[0]    # 5
    action_dim = env.action_space.shape[0]         # 1
    action_max = float(env.action_space.high[0])   # = 1.0

    # networks + targets
    actor  = Actor(state_dim, action_dim, action_max).to(device)
    act_t  = Actor(state_dim, action_dim, action_max).to(device)
    act_t.load_state_dict(actor.state_dict())

    c1 = Critic(state_dim, action_dim).to(device)
    c2 = Critic(state_dim, action_dim).to(device)
    c1_t = Critic(state_dim, action_dim).to(device)
    c2_t = Critic(state_dim, action_dim).to(device)
    c1_t.load_state_dict(c1.state_dict())
    c2_t.load_state_dict(c2.state_dict())

    act_opt = optim.Adam(actor.parameters(),  lr=ACTOR_LR)
    c1_opt  = optim.Adam(c1.parameters(),      lr=CRITIC_LR)
    c2_opt  = optim.Adam(c2.parameters(),      lr=CRITIC_LR)

    buf = ReplayBuffer(REPLAY_SIZE)

    state, _ = env.reset()
    ep_ret, ep_len, ep = 0.0, 0, 0
    best_score = -float("inf")
    pbar = tqdm(range(1, MAX_STEPS+1))

    for t in pbar:
        # explore with Gaussian noise or random actions
        if t < START_STEPS:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                a = actor(torch.as_tensor(state, dtype=torch.float32,
                                          device=device).unsqueeze(0))
                a = a.cpu().numpy()[0]
            action = (a + np.random.normal(0, 0.1, size=action_dim)
                      ).clip(-action_max, action_max)

        next_state, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        buf.push(state, action, reward, next_state, float(done))

        state = next_state
        ep_ret += reward
        ep_len += 1

        if done:
            pbar.set_description(f"Episode {ep:4d} | length {ep_len:4d} | return {ep_ret:8.1f}")
            state, _ = env.reset()
            ep_ret, ep_len, ep = 0.0, 0, ep + 1

        if t >= START_STEPS:
            td3_update(buf, actor, act_t, c1, c2, c1_t, c2_t,
                       act_opt, c1_opt, c2_opt, t)

        if t % EVAL_INTERVAL == 0:
            score = evaluate(actor, n_episodes=20)
            print(f"[eval @ {t:6d}] score (mean-std): {score:7.1f}")
            if score > best_score:
                best_score = score
                torch.save(actor.state_dict(), "td3_cartpole_actor.pth")
                print("  ↳ new best model saved")

    env.close()
    print(f"Training finished.  Best score = {best_score:.1f}")


# ────────────────────────────────────────────────────────────────────────────────
# Quick deterministic evaluation
# ────────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(actor, n_episodes=20):
    env = make_env()
    scores = []
    for _ in range(n_episodes):
        s, _ = env.reset(seed=np.random.randint(0, 1_000_000))
        ep_ret, done = 0.0, False
        while not done:
            a = actor(torch.as_tensor(s, dtype=torch.float32,
                                      device=device).unsqueeze(0))
            a = a.cpu().numpy()[0]
            s, r, term, trunc, _ = env.step(a)
            ep_ret += r
            done = term or trunc
        scores.append(ep_ret)
    env.close()
    mean, std = np.mean(scores), np.std(scores)
    return float(mean - std)      # the assignment’s “score” metric


# ────────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train(seed=args.seed)
