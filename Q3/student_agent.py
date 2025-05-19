import os, numpy as np, torch, torch.nn as nn


# ────────────────────────────────────────────────────────────────────────────────
# Network architecture must mirror train.py
# ────────────────────────────────────────────────────────────────────────────────
def mlp(in_dim, out_dim, h1=400, h2=300, final_tanh=True):
    layers = [ nn.Linear(in_dim, h1), nn.ReLU(),
               nn.Linear(h1, h2), nn.ReLU(),
               nn.Linear(h2, out_dim) ]
    if final_tanh: layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_max):
        super().__init__()
        self.net = mlp(state_dim, action_dim, final_tanh=True)
        self.scale = action_max

    def forward(self, s):
        return self.net(s) * self.scale


# ────────────────────────────────────────────────────────────────────────────────
# Autograder-facing wrapper
# ────────────────────────────────────────────────────────────────────────────────
class Agent(object):
    """
    Deterministic DDPG policy for Humanoid-Walk (67-dim state → 21-dim action).
    """
    def __init__(self):
        self.device = torch.device("cpu")       # leaderboard runs on CPU
        state_dim, action_dim, action_max = 67, 21, 1.0

        self.actor = Actor(state_dim, action_dim, action_max).to(self.device)

        path = os.path.join(os.path.dirname(__file__),
                            "ddpg_humanoid_actor.pth")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                "Trained weight file 'ddpg_humanoid_actor.pth' "
                "not found next to student_agent.py.  Run train.py first."
            )
        self.actor.load_state_dict(
            torch.load(path, map_location=self.device))
        self.actor.eval()

    @torch.no_grad()
    def act(self, observation):
        obs = torch.as_tensor(observation, dtype=torch.float32,
                              device=self.device).unsqueeze(0)
        action = self.actor(obs).cpu().numpy()[0]
        return action.astype(np.float32)

