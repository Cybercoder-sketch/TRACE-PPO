import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical, Normal
from collections import deque
from datetime import datetime

# =====================================================
# 3. Basic Components
# =====================================================
class RunningMeanStd:
    def __init__(self, eps=1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = eps

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)

        if x.size == 0:
            return

        bmean = x.mean()
        bvar = x.var()
        bcount = x.size

        delta = bmean - self.mean
        tot = self.count + bcount

        self.mean += delta * bcount / tot

        m_a = self.var * self.count
        m_b = bvar * bcount
        M2 = m_a + m_b + delta ** 2 * self.count * bcount / tot

        self.var = M2 / tot
        self.count = tot

    def norm(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


class BeliefEncoder(nn.Module):
    def __init__(self, obs_dim, belief_dim=64, n_phases=3):
        super().__init__()

        self.gru = nn.GRUCell(obs_dim, belief_dim)
        self.phase_head = nn.Linear(belief_dim, n_phases)
        self.attack_head = nn.Linear(belief_dim, 2)

        self.belief_dim = belief_dim
        self.n_phases = n_phases

    def forward(self, o_t, b_prev):
        return self.gru(o_t, b_prev)

    def phase_logits(self, b):
        return self.phase_head(b)

    def attack_logits(self, b):
        return self.attack_head(b)

    def attack_prob(self, b):
        return F.softmax(self.attack_logits(b), dim=-1)[..., 1]

    def init_belief(self, batch, device):
        return torch.zeros(batch, self.belief_dim, device=device)


class HybridActor(nn.Module):
    def __init__(
        self,
        belief_dim,
        n_types=3,
        hidden=128,
        n_phases=3,
        logstd_init=-0.5,
        gate_low=0.3,
        gate_high=0.7,
        gate_low_fallback=0.12,
        type_ent_weight=1.5,
        mu_ent_weight=0.5,
    ):
        super().__init__()

        input_dim = belief_dim + n_phases

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        self.type_head = nn.Linear(hidden, n_types)
        self.mu_mean = nn.Linear(hidden, n_types)
        self.mu_logstd = nn.Parameter(torch.zeros(n_types) + logstd_init)

        self.n_phases = n_phases
        self.gate_low = gate_low
        self.gate_high = gate_high
        self.gate_low_fallback = gate_low_fallback

        self.type_ent_weight = type_ent_weight
        self.mu_ent_weight = mu_ent_weight

    def _input(self, b, phase_logits):
        phase_probs = F.softmax(phase_logits.detach(), dim=-1)
        return torch.cat([b, phase_probs], dim=-1)

    def forward(self, b, phase_logits):
        x = self._input(b, phase_logits)
        h = self.trunk(x)

        type_logits = self.type_head(h).clamp(-15.0, 15.0)
        mu_mean = torch.sigmoid(self.mu_mean(h))

        return type_logits, mu_mean

    def _safe_logstd(self, a_type):
        return self.mu_logstd[a_type].clamp(-1.0, 1.0)

    def _apply_gate(
        self,
        type_logits,
        attack_prob,
        gate_active=True,
        a1_fallback_active=False,
    ):
        if not gate_active:
            return type_logits

        ap = attack_prob.detach()
        out = type_logits.clone()

        neg = torch.full_like(out[..., 0], -1e9)

        effective_gate_low = self.gate_low_fallback if a1_fallback_active else self.gate_low

        mask1 = ap < effective_gate_low
        mask2 = ap < self.gate_high

        out[..., 1] = torch.where(mask1, neg, out[..., 1])
        out[..., 2] = torch.where(mask2, neg, out[..., 2])

        return out

    def _entropy(self, type_dist, mu_dist):
        return self.type_ent_weight * type_dist.entropy() + self.mu_ent_weight * mu_dist.entropy()

    def act(
        self,
        b,
        phase_logits,
        attack_prob,
        gate_active=True,
        a1_fallback_active=False,
    ):
        type_logits, mu_mean = self.forward(b, phase_logits)

        type_logits = self._apply_gate(
            type_logits,
            attack_prob,
            gate_active=gate_active,
            a1_fallback_active=a1_fallback_active,
        )

        type_dist = Categorical(logits=type_logits)
        a_type = type_dist.sample()

        idx = a_type.unsqueeze(-1)
        mu_m = mu_mean.gather(-1, idx).squeeze(-1)
        mu_std = self._safe_logstd(a_type).exp()

        mu_dist = Normal(mu_m, mu_std)
        a_mu = mu_dist.sample().clamp(1e-4, 1.0 - 1e-4)

        logp = type_dist.log_prob(a_type) + mu_dist.log_prob(a_mu)
        ent = self._entropy(type_dist, mu_dist)

        return a_type, a_mu, logp, ent

    def evaluate(
        self,
        b,
        phase_logits,
        attack_prob,
        a_type,
        a_mu,
        gate_active=True,
        gate_attack_prob=None,
        a1_fallback_active=False,
    ):
        type_logits, mu_mean = self.forward(b, phase_logits)

        if gate_attack_prob is None:
            gate_attack_prob = attack_prob

        type_logits = self._apply_gate(
            type_logits,
            gate_attack_prob,
            gate_active=gate_active,
            a1_fallback_active=a1_fallback_active,
        )

        type_dist = Categorical(logits=type_logits)

        idx = a_type.unsqueeze(-1)
        mu_m = mu_mean.gather(-1, idx).squeeze(-1)
        mu_std = self._safe_logstd(a_type).exp()

        mu_dist = Normal(mu_m, mu_std)

        logp = type_dist.log_prob(a_type) + mu_dist.log_prob(a_mu.clamp(1e-4, 1.0 - 1e-4))
        ent = self._entropy(type_dist, mu_dist)

        return logp, ent


class QuantileCritic(nn.Module):
    def __init__(self, belief_dim, n_quant=32, hidden=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(belief_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_quant),
        )

        self.n_quant = n_quant
        self.register_buffer("taus", (torch.arange(n_quant).float() + 0.5) / n_quant)

    def forward(self, b):
        return self.net(b)


class ScalarCritic(nn.Module):
    def __init__(self, belief_dim, hidden=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(belief_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, b):
        return self.net(b).squeeze(-1)


