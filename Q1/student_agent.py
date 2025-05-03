import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


# Network definition must match training script
def mlp(input_dim, output_dim, hidden=256, final_tanh=True):
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


class Agent(object):
    def __init__(self):
        self.device = torch.device("cpu")        # leaderboard requires CPU
        dummy_env = gym.make("Pendulum-v1")
        state_dim = dummy_env.observation_space.shape[0]
        action_dim = dummy_env.action_space.shape[0]
        action_max = dummy_env.action_space.high[0]

        self.actor = Actor(state_dim, action_dim, action_max).to(self.device)

        # Load weights saved by train.py
        model_path = os.path.join(os.path.dirname(__file__),
                                  "td3_pendulum_actor.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Trained model not found at {model_path}. "
                "Make sure you ran train.py and copied the .pth file."
            )

        self.actor.load_state_dict(
            torch.load(model_path, map_location=self.device))
        self.actor.eval()      # switch to inference mode
        dummy_env.close()

    @torch.no_grad()
    def act(self, observation):
        obs = torch.as_tensor(observation,
                              dtype=torch.float32,
                              device=self.device).unsqueeze(0)
        action = self.actor(obs).cpu().numpy()[0]
        return action.astype(np.float32)
