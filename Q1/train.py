import argparse
import os
import random
from collections import deque, namedtuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ────────────────────────────────────────────────────────────────────────────────
# Hyper-parameters (override with CLI flags if you wish)
# ────────────────────────────────────────────────────────────────────────────────
HIDDEN_SIZE = 256
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
GAMMA = 0.99
TAU = 0.005                 # Polyak averaging factor
POLICY_DELAY = 2
POLICY_NOISE = 0.2
NOISE_CLIP = 0.5
REPLAY_SIZE = int(1e6)
BATCH_SIZE = 256
START_STEPS = 5000          # warm-up with random actions
MAX_STEPS = 250_000
EVAL_INTERVAL = 10_000


# ────────────────────────────────────────────────────────────────────────────────
# Utilities
# ────────────────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else exit(1))
Transition = namedtuple("Transition",
                        "state action reward next_state done")

class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def push(self, *args):
        self.buf.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        t = Transition(*zip(*batch))
        states = torch.as_tensor(t.state, dtype=torch.float32, device=device)
        actions = torch.as_tensor(t.action, dtype=torch.float32, device=device)
        rewards = torch.as_tensor(t.reward, dtype=torch.float32,
                                  device=device).unsqueeze(1)
        next_states = torch.as_tensor(t.next_state, dtype=torch.float32,
                                      device=device)
        dones = torch.as_tensor(t.done, dtype=torch.float32,
                                device=device).unsqueeze(1)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buf)


# ────────────────────────────────────────────────────────────────────────────────
# Networks
# ────────────────────────────────────────────────────────────────────────────────
def mlp(input_dim, output_dim, hidden=HIDDEN_SIZE, final_tanh=False):
    layers = [
        nn.Linear(input_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, output_dim)
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
    """Q(s,a) network; two identical copies are kept for TD3."""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = mlp(state_dim + action_dim, 1)

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))


# ────────────────────────────────────────────────────────────────────────────────
# TD3 update step
# ────────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def soft_update(target, src, tau):
    for tp, p in zip(target.parameters(), src.parameters()):
        tp.data.mul_(1 - tau).add_(p.data, alpha=tau)


def td3_update(replay, actor, actor_target, c1, c2,
               c1_target, c2_target, actor_opt, c1_opt, c2_opt,
               total_it):
    if len(replay) < BATCH_SIZE:
        return

    s, a, r, s2, d = replay.sample(BATCH_SIZE)

    # Target policy smoothing
    noise = (torch.randn_like(a) * POLICY_NOISE).clamp(-NOISE_CLIP, NOISE_CLIP)
    a2 = (actor_target(s2) + noise).clamp(-2.0, 2.0)

    with torch.no_grad():
        q1_t = c1_target(s2, a2)
        q2_t = c2_target(s2, a2)
        q_t = torch.min(q1_t, q2_t)
        y = r + GAMMA * (1 - d) * q_t

    # Critic loss
    q1 = c1(s, a)
    q2 = c2(s, a)
    critic_loss = nn.functional.mse_loss(q1, y) + nn.functional.mse_loss(q2, y)

    c1_opt.zero_grad()
    c2_opt.zero_grad()
    critic_loss.backward()
    c1_opt.step()
    c2_opt.step()

    # Policy update (delayed)
    if total_it % POLICY_DELAY == 0:
        act_out = actor(s)
        actor_loss = -c1(s, act_out).mean()
        actor_opt.zero_grad()
        actor_loss.backward()
        actor_opt.step()

        # Target networks
        soft_update(actor_target, actor, TAU)
        soft_update(c1_target, c1, TAU)
        soft_update(c2_target, c2, TAU)


# ────────────────────────────────────────────────────────────────────────────────
# Training loop
# ────────────────────────────────────────────────────────────────────────────────
def train(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make("Pendulum-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_max = env.action_space.high[0]

    actor = Actor(state_dim, action_dim, action_max).to(device)
    actor_target = Actor(state_dim, action_dim, action_max).to(device)
    actor_target.load_state_dict(actor.state_dict())

    critic1 = Critic(state_dim, action_dim).to(device)
    critic2 = Critic(state_dim, action_dim).to(device)
    critic1_target = Critic(state_dim, action_dim).to(device)
    critic2_target = Critic(state_dim, action_dim).to(device)
    critic1_target.load_state_dict(critic1.state_dict())
    critic2_target.load_state_dict(critic2.state_dict())

    actor_opt = optim.Adam(actor.parameters(), lr=ACTOR_LR)
    critic1_opt = optim.Adam(critic1.parameters(), lr=CRITIC_LR)
    critic2_opt = optim.Adam(critic2.parameters(), lr=CRITIC_LR)

    replay = ReplayBuffer(REPLAY_SIZE)

    state, _ = env.reset(seed=seed)
    episode_reward, episode_steps = 0.0, 0
    ep = 0
    best_eval = -float("inf")

    for t in range(1, MAX_STEPS + 1):
        # Exploration vs. policy action
        if t < START_STEPS:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                a = actor(torch.as_tensor(state, dtype=torch.float32,
                                          device=device).unsqueeze(0))
                a = a.cpu().numpy()[0]
            # exploration noise
            action = (a + np.random.normal(0, 0.1, size=action_dim)
                      ).clip(-action_max, action_max)

        next_state, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        replay.push(state, action, reward, next_state, float(done))

        state = next_state
        episode_reward += reward
        episode_steps += 1

        if done:
            print(f"Episode {ep:4d} | steps {episode_steps:3d} | "
                  f"return {episode_reward:7.1f}")
            state, _ = env.reset()
            episode_reward, episode_steps = 0.0, 0
            ep += 1

        # TD3 update
        if t >= START_STEPS:
            td3_update(replay, actor, actor_target,
                       critic1, critic2,
                       critic1_target, critic2_target,
                       actor_opt, critic1_opt, critic2_opt,
                       total_it=t)

        # Quick evaluation (no exploration) every EVAL_INTERVAL steps
        if t % EVAL_INTERVAL == 0:
            mean_r = evaluate(actor, env, n_episodes=10)
            print(f"[eval @ step {t:6d}] average return: {mean_r:7.1f}")
            if mean_r > best_eval:
                best_eval = mean_r
                save_path = "td3_pendulum_actor.pth"
                torch.save(actor.state_dict(), save_path)
                print(f"  ↳ saved better model to {save_path}")

    env.close()
    print(f"Training finished. Best eval return = {best_eval:.1f}")


@torch.no_grad()
def evaluate(actor, env, n_episodes=10):
    returns = []
    for _ in range(n_episodes):
        s, _ = env.reset(seed=np.random.randint(0, 1_000_000))
        ep_r = 0.0
        done = False
        while not done:
            a = actor(torch.as_tensor(s, dtype=torch.float32,
                                      device=device).unsqueeze(0))
            a = a.cpu().numpy()[0]
            s, r, term, trunc, _ = env.step(a)
            done = term or trunc
            ep_r += r
        returns.append(ep_r)
    return float(np.mean(returns))


# ────────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train(seed=args.seed)
