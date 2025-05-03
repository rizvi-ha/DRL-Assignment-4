import os
import numpy as np
import torch
import torch.nn as nn

def mlp(inp,out,hid=400,tanh=True):
    layers=[nn.Linear(inp,hid),nn.ReLU(),
            nn.Linear(hid,hid),nn.ReLU(),
            nn.Linear(hid,out)]
    if tanh: layers.append(nn.Tanh())
    return nn.Sequential(*layers)

class Actor(nn.Module):
    def __init__(self,sdim,adim,amax):
        super().__init__(); self.body=mlp(sdim,adim,tanh=True); self.amax=amax
    def forward(self,s): return self.body(s)*self.amax

class Agent(object):
    """Deterministic TD3 policy for Humanoid-Walk (state obs)."""
    def __init__(self):
        self.device=torch.device("cpu")        # leaderboard runs on CPU
        sdim,adim,amax=67,21,1.0    
        self.actor=Actor(sdim,adim,amax).to(self.device)
        path=os.path.join(os.path.dirname(__file__),"td3_humanoid_actor.pth")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Weight file missing at {path}")
        self.actor.load_state_dict(torch.load(path,map_location=self.device))
        self.actor.eval()

    @torch.no_grad()
    def act(self,observation):
        obs=torch.as_tensor(observation,dtype=torch.float32,
                            device=self.device).unsqueeze(0)
        action=self.actor(obs).cpu().numpy()[0]
        return action.astype(np.float32)
