import os
import numpy as np
import torch
import torch.nn as nn

# Network definition (must match train.py)
def mlp(in_dim, out_dim, hidden=256, final_tanh=True):
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


class Agent(object):
    def __init__(self):
        self.device = torch.device("cpu")          # leaderboard runs on CPU

        state_dim  = 5        # cart position, pole angle, & derivatives
        action_dim = 1
        action_max = 1.0

        self.actor = Actor(state_dim, action_dim, action_max).to(self.device)

        weight_path = os.path.join(os.path.dirname(__file__),
                                   "td3_cartpole_actor.pth")
        if not os.path.isfile(weight_path):
            raise FileNotFoundError(
                f"Weight file not found at {weight_path}. "
                "Did you run train.py and copy the .pth here?"
            )

        self.actor.load_state_dict(
            torch.load(weight_path, map_location=self.device))
        self.actor.eval()

    @torch.no_grad()
    def act(self, observation):
        obs = torch.as_tensor(observation, dtype=torch.float32,
                              device=self.device).unsqueeze(0)
        action = self.actor(obs).cpu().numpy()[0]
        return action.astype(np.float32)
