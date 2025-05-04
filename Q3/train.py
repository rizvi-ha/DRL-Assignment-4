import argparse, os, random
from collections import deque, namedtuple

import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from dmc import make_dmc_env            # provided by assignment repo
from tqdm import tqdm

# ── hyper-params ────────────────────────────────────────────────────────────────
HIDDEN        = 400
ACTOR_LR      = 6e-4
CRITIC_LR     = 1e-3
GAMMA         = 0.99
TAU           = 0.005
POLICY_DELAY  = 2
POLICY_NOISE  = 0.2
NOISE_CLIP    = 0.5
REPLAY_SIZE   = int(2e6)
BATCH_SIZE    = 512
START_STEPS   = 25_000
MAX_STEPS     = 1_000_000
EVAL_INTERVAL = 50_000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T = namedtuple("Transition", "s a r s2 d")

# ── replay buffer ───────────────────────────────────────────────────────────────
class ReplayBuf:
    def __init__(self, cap): self.buf=deque(maxlen=cap)
    def push(self,*x): self.buf.append(T(*x))
    def __len__(self): return len(self.buf)
    def sample(self,k):
        b=random.sample(self.buf,k); t=T(*zip(*b))
        s  =torch.as_tensor(np.stack(t.s),  dtype=torch.float32,device=device)
        a  =torch.as_tensor(np.stack(t.a),  dtype=torch.float32,device=device)
        r  =torch.as_tensor(np.array(t.r),  dtype=torch.float32,device=device).unsqueeze(1)
        s2 =torch.as_tensor(np.stack(t.s2), dtype=torch.float32,device=device)
        d  =torch.as_tensor(np.array(t.d),  dtype=torch.float32,device=device).unsqueeze(1)
        return s,a,r,s2,d

# ── networks ────────────────────────────────────────────────────────────────────
def mlp(inp,out,hidden=HIDDEN,tanh=False):
    layers=[nn.Linear(inp,hidden),nn.ReLU(),
            nn.Linear(hidden,hidden),nn.ReLU(),
            nn.Linear(hidden,out)]
    if tanh: layers.append(nn.Tanh())
    return nn.Sequential(*layers)

class Actor(nn.Module):
    def __init__(self,sdim,adim,amax):
        super().__init__()
        self.body=mlp(sdim,adim,tanh=True); self.amax=amax
    def forward(self,s): return self.body(s)*self.amax

class Critic(nn.Module):
    def __init__(self,sdim,adim):
        super().__init__(); self.q=mlp(sdim+adim,1)
    def forward(self,s,a): return self.q(torch.cat([s,a],dim=-1))

# ── utils ───────────────────────────────────────────────────────────────────────
@torch.no_grad()
def polyak(trg,src,tau): 
    for tp,p in zip(trg.parameters(),src.parameters()):
        tp.data.mul_(1-tau).add_(p.data,alpha=tau)

def td3_step(buf,act,act_t,q1,q2,q1_t,q2_t,opt_a,opt_q1,opt_q2,step):
    if len(buf)<BATCH_SIZE: return
    s,a,r,s2,d=buf.sample(BATCH_SIZE)
    noise=(torch.randn_like(a)*POLICY_NOISE).clamp(-NOISE_CLIP,NOISE_CLIP)
    a2=(act_t(s2)+noise).clamp(-1.0,1.0)
    with torch.no_grad():
        y=r+GAMMA*(1-d)*torch.min(q1_t(s2,a2),q2_t(s2,a2))
    # critics
    loss_q=nn.functional.mse_loss(q1(s,a),y)+nn.functional.mse_loss(q2(s,a),y)
    opt_q1.zero_grad(); opt_q2.zero_grad(); loss_q.backward(); opt_q1.step(); opt_q2.step()
    # delayed actor + targets
    if step%POLICY_DELAY==0:
        loss_a=-q1(s,act(s)).mean()
        opt_a.zero_grad(); loss_a.backward(); opt_a.step()
        polyak(act_t,act,TAU); polyak(q1_t,q1,TAU); polyak(q2_t,q2,TAU)

# ── environment helpers ────────────────────────────────────────────────────────
def make_env(seed=0):
    env=make_dmc_env("humanoid-walk",
                     seed,
                     flatten=True,use_pixels=False)
    return env

# ── evaluation (mean - std) ────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(actor,eps=10):
    env=make_env(); scores=[]
    for _ in range(eps):
        s,_=env.reset(seed=np.random.randint(0,1)); ep_r=0; done=False
        while not done:
            a=actor(torch.as_tensor(s,dtype=torch.float32,device=device).unsqueeze(0))
            s,r,term,trunc,_=env.step(a.cpu().numpy()[0]); ep_r+=r; done=term or trunc
        scores.append(ep_r)
    env.close(); return float(np.mean(scores)-np.std(scores))

# ── main training loop ─────────────────────────────────────────────────────────
def train(seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env=make_env(seed)
    sdim=env.observation_space.shape[0]      # 67  :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}
    adim=env.action_space.shape[0]           # 21  :contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3}
    amax=float(env.action_space.high[0])
    actor=Actor(sdim,adim,amax).to(device);  act_t=Actor(sdim,adim,amax).to(device)
    q1=Critic(sdim,adim).to(device); q2=Critic(sdim,adim).to(device)
    q1_t=Critic(sdim,adim).to(device); q2_t=Critic(sdim,adim).to(device)
    act_t.load_state_dict(actor.state_dict()); q1_t.load_state_dict(q1.state_dict()); q2_t.load_state_dict(q2.state_dict())
    opt_a=optim.Adam(actor.parameters(),ACTOR_LR)
    opt_q1=optim.Adam(q1.parameters(),CRITIC_LR); opt_q2=optim.Adam(q2.parameters(),CRITIC_LR)
    buf=ReplayBuf(REPLAY_SIZE)

    s,_=env.reset(); ep_r=0; ep_len=0; ep=0; best=-float("inf")
    pbar = tqdm(range(1,MAX_STEPS+1))
    for t in pbar:
        if t<START_STEPS: a=env.action_space.sample()
        else:
            with torch.no_grad():
                a=actor(torch.as_tensor(s,dtype=torch.float32,device=device).unsqueeze(0))
                a=a.cpu().numpy()[0]
            a=(a+np.random.normal(0,0.1,adim)).clip(-amax,amax)
        s2,r,term,trunc,_=env.step(a); done=term or trunc
        buf.push(s,a,r,s2,float(done)); s=s2; ep_r+=r; ep_len+=1
        if done:
            pbar.set_description(f"Ep {ep:4d} | len {ep_len:4d} | return {ep_r:6.1f}")
            s,_=env.reset(); ep_r=0; ep_len=0; ep+=1
        if t>=START_STEPS:
            td3_step(buf,actor,act_t,q1,q2,q1_t,q2_t,opt_a,opt_q1,opt_q2,t)
        if t%EVAL_INTERVAL==0:
            score=evaluate(actor,eps=20)
            print(f"[eval {t//1000:4d}k] score {score:7.1f}")
            if score>best:
                best=score
                torch.save(actor.state_dict(),"td3_humanoid_actor.pth")
                print("  ↳ new best saved")
    env.close(); print(f"Finished; best score {best:.1f}")

# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--seed",type=int,default=0)
    train(**vars(p.parse_args()))
