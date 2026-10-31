"""Step 4: RaceDecider-XS, the on-board neural decision model.

Inputs are the 90 step-2 features, split into groups:
  ego   ego_* + trk_* + hist_* + prev_*  → MLP
  scan  scan_* (54 sectors)              → small 1-D conv over angle
  opps  opp0_* (a set; one slot today)   → DeepSets: φ per opponent, masked
                                            sum, so more opponents need no retraining
→ concatenated → trunk → 4 heads:
  state     6-way logits (the decision)
  collide   P(collision within 1 s)          ┐ auxiliary: shape the representation
  progress  raceline metres in the next 2 s  │ and give a risk / value signal the
  pass      P(completing a pass within 3 s)  ┘ fallback logic can use later
"""
import numpy as np
import torch
import torch.nn as nn

from features import STATES, FeatureExtractor

NAMES = FeatureExtractor.names()
SCAN = [n for n in NAMES if n.startswith("scan_")]
OPP = [n for n in NAMES if n.startswith("opp0_")]
EGO = [n for n in NAMES if n not in SCAN and n not in OPP]
PREV = [n for n in EGO if n.startswith("prev_")]


def mlp(i, h, o, n=2):
    layers, d = [], i
    for _ in range(n - 1):
        layers += [nn.Linear(d, h), nn.SiLU()]
        d = h
    return nn.Sequential(*layers, nn.Linear(d, o))


class RaceDecider(nn.Module):
    def __init__(self, width=96):
        super().__init__()
        self.width = width
        self.ego = mlp(len(EGO), width, width)
        self.scan = nn.Sequential(                        # 54 sectors → 16 ch × 7
            nn.Conv1d(1, 8, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv1d(8, 16, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv1d(16, 16, 3, stride=2, padding=1), nn.SiLU(), nn.Flatten(),
            nn.Linear(16 * 7, width))
        self.phi = mlp(len(OPP) - 1, width, width)        # per opponent; the present flag is the mask
        self.rho = mlp(width, width, width)
        self.trunk = mlp(3 * width, 2 * width, width)
        self.head_state = nn.Linear(width, len(STATES))
        self.head_aux = nn.Linear(width, 3)               # collide logit, progress, pass logit
        # normalisation stats, set from the training set and saved with the weights
        self.register_buffer("mu", torch.zeros(len(NAMES)))
        self.register_buffer("sd", torch.ones(len(NAMES)))
        self.idx = {k: [NAMES.index(n) for n in v] for k, v in (("ego", EGO), ("scan", SCAN), ("opp", OPP))}

    def forward(self, x):
        """x: (B, 90) raw features in FeatureExtractor.names() order."""
        z = (x - self.mu) / self.sd
        e = self.ego(z[:, self.idx["ego"]])
        s = self.scan(z[:, self.idx["scan"]].unsqueeze(1))
        o = z[:, self.idx["opp"]]
        mask = x[:, self.idx["opp"][:1]]                  # raw present flag (0/1), (B, 1)
        # (B, N=1, F) set of opponents; masked sum keeps it permutation-invariant
        o = (self.phi(o[:, 1:]) * mask).unsqueeze(1).sum(1)
        h = self.trunk(torch.cat([e, s, self.rho(o)], 1))
        aux = self.head_aux(h)
        return self.head_state(h), aux[:, 0], aux[:, 1], aux[:, 2]


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = RaceDecider(ck["width"])
    m.load_state_dict(ck["model"])
    m.eval()
    return m


def make_decider(path):
    """Closed-loop callable(features) → state, for generate.run_episode."""
    torch.set_num_threads(1)
    m = load(path)

    @torch.no_grad()
    def decide(feat):
        return STATES[int(m(torch.from_numpy(np.asarray(feat, np.float32))[None])[0].argmax())]
    return decide
