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


from models import BeliefEncoder, HybridActor, QuantileCritic, ScalarCritic, RunningMeanStd
from utils import PIDLagrangian, empirical_cvar_weighted
from config import SAFE_CKPT_NAME
# =====================================================
# 6. TRACE-PPO V7.2.1 Trainer
# =====================================================
class TRACE_PPO:
    def __init__(
        self,
        obs_dim,
        n_types=3,
        belief_dim=64,
        lr=2e-4,
        gamma=0.99,
        clip_eps=0.18,
        alpha=0.85,
        c_limit_benign=5.0,
        c_limit_attack=20.0,
        attack_rate=0.55,
        gate_low=0.3,
        gate_high=0.7,
        gate_low_fallback=0.12,
        gate_warmup_eps=50,
        pid_start_eps=60,
        pid_ramp_eps=60,
        device="cpu",
    ):
        self.device = device

        self.encoder = BeliefEncoder(obs_dim, belief_dim).to(device)

        self.actor = HybridActor(
            belief_dim,
            n_types,
            gate_low=gate_low,
            gate_high=gate_high,
            gate_low_fallback=gate_low_fallback,
        ).to(device)

        self.critic_r = ScalarCritic(belief_dim).to(device)
        self.critic_c = QuantileCritic(belief_dim, n_quant=32).to(device)

        params = (
            list(self.encoder.parameters())
            + list(self.actor.parameters())
            + list(self.critic_r.parameters())
            + list(self.critic_c.parameters())
        )

        self.opt = torch.optim.Adam(params, lr=lr)
        self.base_lr = lr

        self.gamma = gamma
        self.clip_eps = clip_eps
        self.alpha = alpha

        self.attack_rate = attack_rate
        self.c_limit_benign = c_limit_benign
        self.c_limit_attack = c_limit_attack

        self.w_b = 1.0 - attack_rate
        self.w_a = attack_rate
        self.c_limit_joint = self.w_b * c_limit_benign + self.w_a * c_limit_attack

        self.pid = PIDLagrangian(
            kp=0.5,
            ki=0.02,
            kd=0.3,
            lam_max=30.0,
            leak=0.985,
            rate_limit=2.0,
            dead_band_frac=0.10,
        )

        self.cost_buf_benign = deque(maxlen=80)
        self.cost_buf_mal = deque(maxlen=80)

        self.cost_recent_benign = deque(maxlen=10)
        self.cost_recent_mal = deque(maxlen=10)

        self.main_decay_benign = 0.92
        self.main_decay_mal = 0.96
        self.recent_mix_benign = 0.85
        self.recent_mix_mal = 0.65

        self.cost_norm = RunningMeanStd()

        self.episode_count = 0

        self.kl_target = 0.02
        self.kl_beta = 1.0
        self.cvar_k = int(self.alpha * 32)

        self.ent_coef = 0.02
        self.ent_coef_floor = 0.015
        self.ent_coef_ceil = 0.08

        self.gate_warmup_eps = gate_warmup_eps
        self.pid_start_eps = pid_start_eps
        self.pid_ramp_eps = pid_ramp_eps

        self.best_R_avg = -1e9
        self.best_state = None

        self.best_safe_score = -1e9
        self.best_safe_ep = -1
        self.safe_ckpt_path = os.path.join(os.getcwd(), SAFE_CKPT_NAME)

        self.recent_R = deque(maxlen=50)
        self.recent_C = deque(maxlen=50)
        self.recent_C_b = deque(maxlen=30)
        self.recent_C_a = deque(maxlen=30)
        self.recent_Vmin_b = deque(maxlen=30)
        self.recent_Vmin_a = deque(maxlen=30)

        self.recent_a2_b = deque(maxlen=30)
        self.recent_a2_a = deque(maxlen=30)

        self.kl_zero_streak = 0
        self.last_rollback_ep = -10**6
        self.underperf_streak = 0
        self.rollback_cooldown = 200
        self.underperf_threshold = 15.0

        self.dead_zone_streak = 0
        self.request_explore_epoch = False
        self.low_ent_streak = 0

        self.sat_streak = 0

    def _safe_normalize(self, x):
        sd = x.std()

        if sd > 1e-8:
            return (x - x.mean()) / (sd + 1e-8)

        return x - x.mean()

    def _encode_sequence(self, obs_t):
        T = obs_t.shape[0]
        b = self.encoder.init_belief(1, self.device)
        beliefs = []

        for t in range(T):
            o = obs_t[t].unsqueeze(0)
            b = self.encoder(o, b)
            beliefs.append(b.squeeze(0))

        b_seq = torch.stack(beliefs, dim=0)
        phase_logits = self.encoder.phase_logits(b_seq)
        attack_prob = self.encoder.attack_prob(b_seq)

        return b_seq, phase_logits, attack_prob

    def gae(self, vals, rs, ds, lam=0.95):
        adv = np.zeros_like(rs)
        g = 0.0
        nv = 0.0

        for t in reversed(range(len(rs))):
            delta = rs[t] + self.gamma * nv * (1.0 - ds[t]) - vals[t]
            g = delta + self.gamma * lam * (1.0 - ds[t]) * g
            adv[t] = g
            nv = vals[t]

        return adv

    def _save_snapshot(self):
        self.best_state = {
            "enc": {k: v.detach().clone() for k, v in self.encoder.state_dict().items()},
            "act": {k: v.detach().clone() for k, v in self.actor.state_dict().items()},
            "cr": {k: v.detach().clone() for k, v in self.critic_r.state_dict().items()},
            "cc": {k: v.detach().clone() for k, v in self.critic_c.state_dict().items()},
        }

    def _load_snapshot(self):
        if self.best_state is None:
            return

        self.encoder.load_state_dict(self.best_state["enc"])
        self.actor.load_state_dict(self.best_state["act"])
        self.critic_r.load_state_dict(self.best_state["cr"])
        self.critic_c.load_state_dict(self.best_state["cc"])

    def _state_dict_for_ckpt(self):
        return {
            "encoder": self.encoder.state_dict(),
            "actor": self.actor.state_dict(),
            "critic_r": self.critic_r.state_dict(),
            "critic_c": self.critic_c.state_dict(),
            "optimizer": self.opt.state_dict(),
            "pid": {
                "lam": self.pid.lam,
                "integral": self.pid.integral,
                "prev_err": self.pid.prev_err,
            },
            "episode_count": self.episode_count,
            "best_safe_score": self.best_safe_score,
            "best_safe_ep": self.best_safe_ep,
            "training_mode": "unsupervised_label_blind",
        }

    def _cvar_with_decay(self, values, decay=0.96):
        vals = list(values)
        n = len(vals)

        if n == 0:
            return 0.0

        arr = np.asarray(vals, dtype=np.float64)

        if n == 1:
            return float(arr[0])

        weights = np.array([decay ** (n - 1 - i) for i in range(n)], dtype=np.float64)
        weights = np.maximum(weights, 1e-6)

        return empirical_cvar_weighted(arr, weights=weights, alpha=self.alpha)

    def _bucket_cvars(self):
        main_b = list(self.cost_buf_benign)
        main_m = list(self.cost_buf_mal)

        recent_b = list(self.cost_recent_benign)
        recent_m = list(self.cost_recent_mal)

        main_cvar_b = self._cvar_with_decay(main_b, decay=self.main_decay_benign)
        main_cvar_m = self._cvar_with_decay(main_m, decay=self.main_decay_mal)

        recent_cvar_b = self._cvar_with_decay(recent_b, decay=0.98)
        recent_cvar_m = self._cvar_with_decay(recent_m, decay=0.98)

        if len(recent_b) > 0:
            cvar_b = (1.0 - self.recent_mix_benign) * main_cvar_b + self.recent_mix_benign * recent_cvar_b
        else:
            cvar_b = main_cvar_b

        if len(recent_m) > 0:
            cvar_a = (1.0 - self.recent_mix_mal) * main_cvar_m + self.recent_mix_mal * recent_cvar_m
        else:
            cvar_a = main_cvar_m

        return float(cvar_b), float(cvar_a)

    def joint_cvar(self, cvar_b, cvar_a):
        return self.w_b * cvar_b + self.w_a * cvar_a

    def clear_cost_buffers(self):
        self.cost_buf_benign.clear()
        self.cost_buf_mal.clear()
        self.cost_recent_benign.clear()
        self.cost_recent_mal.clear()

    def _compute_phase_class_weights(self, phase_labels_np):
        counts = np.bincount(phase_labels_np.astype(np.int64), minlength=3).astype(np.float32) + 1.0
        N = float(phase_labels_np.size)

        w = N / (3.0 * counts)
        w = np.clip(w, 0.3, 5.0)

        return torch.as_tensor(w, dtype=torch.float32, device=self.device)

    def _compute_attack_class_weights(self, phase_labels_np):
        is_attack = (phase_labels_np > 0).astype(np.int64)

        counts = np.bincount(is_attack, minlength=2).astype(np.float32) + 1.0
        N = float(is_attack.size)

        w = N / (2.0 * counts)
        w = np.clip(w, 0.3, 5.0)

        return (
            torch.as_tensor(w, dtype=torch.float32, device=self.device),
            torch.as_tensor(is_attack, dtype=torch.long, device=self.device),
        )

    def _apply_sanity_gate(self, enabled=True):
        action = "none"

        if not enabled:
            self.sat_streak = 0
            return action

        if self.pid.lam >= 0.95 * self.pid.lam_max:
            self.sat_streak += 1
        else:
            self.sat_streak = max(0, self.sat_streak - 2)

        if self.sat_streak >= 40:
            self.pid.clear(hard=True)
            self.sat_streak = 0
            action = "flush"
        elif self.sat_streak >= 20 and self.sat_streak % 5 == 0:
            self.pid.lam *= 0.85
            self.pid.integral *= 0.5
            action = "soft"

        return action

    def _update_pid_lambda(self, cvar_joint):
        if self.episode_count < self.pid_start_eps:
            return 0.0, 0.0, "none"

        lam_raw = self.pid.update(cvar_joint, self.c_limit_joint)

        ramp = min(
            1.0,
            max(
                0.0,
                (self.episode_count - self.pid_start_eps) / max(1, self.pid_ramp_eps),
            ),
        )

        lam_eff = ramp * lam_raw

        sanity_enabled = self.episode_count >= (self.pid_start_eps + self.pid_ramp_eps)
        sanity_action = self._apply_sanity_gate(enabled=sanity_enabled)

        return lam_eff, lam_raw, sanity_action

    def risk_fallback_active(self):
        if self.episode_count < self.gate_warmup_eps:
            return False

        cvar_b, cvar_a = self._bucket_cvars()
        cvar_j = self.joint_cvar(cvar_b, cvar_a)

        recent_m = list(self.cost_recent_mal)
        recent_m_mean = float(np.mean(recent_m)) if recent_m else 0.0
        recent_m_max = float(np.max(recent_m)) if recent_m else 0.0

        cond_recent = recent_m_mean > 0.45 * self.c_limit_attack or recent_m_max > 0.75 * self.c_limit_attack
        cond_cvar_a = cvar_a > 0.70 * self.c_limit_attack
        cond_joint = cvar_j > 0.80 * self.c_limit_joint

        return bool(cond_recent or cond_cvar_a or cond_joint)

    def update(self, batch, phase_labels, epochs=18):
        last_kl = 0.0
        last_ent = 0.0
        last_aux_phase = 0.0
        last_aux_attack = 0.0
        last_gate_block_rate = 0.0

        obs_t = torch.as_tensor(np.asarray(batch["obs"]), dtype=torch.float32, device=self.device)
        gate_ap_old = torch.as_tensor(np.asarray(batch["attack_prob"]), dtype=torch.float32, device=self.device)

        at = torch.as_tensor(batch["a_type"], dtype=torch.long, device=self.device)
        am = torch.as_tensor(batch["a_mu"], dtype=torch.float32, device=self.device)
        lp0 = torch.as_tensor(batch["logp_old"], dtype=torch.float32, device=self.device)

        r = np.asarray(batch["rewards"], dtype=np.float32)
        c = np.asarray(batch["costs"], dtype=np.float32)
        d = np.asarray(batch["dones"], dtype=np.float32)

        phase_np = np.asarray(phase_labels, dtype=np.int64)
        phase_np = np.clip(phase_np, 0, 2)

        phase_weights = self._compute_phase_class_weights(phase_np)
        pl_t = torch.as_tensor(phase_np, dtype=torch.long, device=self.device)

        attack_weights, attack_t = self._compute_attack_class_weights(phase_np)

        gate_active = self.episode_count >= self.gate_warmup_eps
        a1_fallback_active = bool(batch.get("a1_fallback_active", False))

        with torch.no_grad():
            effective_gate_low = self.actor.gate_low_fallback if a1_fallback_active else self.actor.gate_low
            blk1 = (gate_ap_old < effective_gate_low).float().mean().item()
            blk2 = (gate_ap_old < self.actor.gate_high).float().mean().item()
            last_gate_block_rate = 0.5 * (blk1 + blk2) if gate_active else 0.0

        cvar_b, cvar_a = self._bucket_cvars()
        cvar_joint = self.joint_cvar(cvar_b, cvar_a)

        lam, lam_raw, sanity_action = self._update_pid_lambda(cvar_joint)

        self.cost_norm.update(c)
        c_norm = self.cost_norm.norm(c).astype(np.float32)

        with torch.no_grad():
            b0, _, _ = self._encode_sequence(obs_t)

            vR = self.critic_r(b0).detach().cpu().numpy()
            qC = self.critic_c(b0)

            vC_mean = qC.mean(-1).detach().cpu().numpy()
            vC_tail = qC[..., self.cvar_k:].mean(-1).detach().cpu().numpy()

        adv_r = self.gae(vR, r, d)
        adv_c_mean = self.gae(vC_mean, c_norm, d)
        adv_c_tail = self.gae(vC_tail, c_norm, d)

        ret_r_t = torch.as_tensor(adv_r + vR, dtype=torch.float32, device=self.device)
        ret_c_mean_t = torch.as_tensor(adv_c_mean + vC_mean, dtype=torch.float32, device=self.device)

        adv_c_blend = 0.3 * adv_c_mean + 0.7 * adv_c_tail

        adv_r_t = torch.as_tensor(adv_r, dtype=torch.float32, device=self.device)
        adv_c_blend_t = torch.as_tensor(adv_c_blend, dtype=torch.float32, device=self.device)

        adv_eff = adv_r_t - lam * adv_c_blend_t
        adv_eff = self._safe_normalize(adv_eff)

        # Self-supervised auxiliary loss weight.
        aux_weight = 0.25

        do_explore_epoch = self.request_explore_epoch
        self.request_explore_epoch = False

        for epoch in range(epochs):
            b_cur, pl_cur, ap_cur = self._encode_sequence(obs_t)

            logp, ent = self.actor.evaluate(
                b_cur,
                pl_cur,
                ap_cur,
                at,
                am,
                gate_active=gate_active,
                gate_attack_prob=gate_ap_old,
                a1_fallback_active=a1_fallback_active,
            )

            log_ratio = logp - lp0
            approx_kl = ((torch.exp(log_ratio) - 1.0) - log_ratio).mean().item()

            last_kl = approx_kl
            last_ent = ent.mean().item()

            if approx_kl > self.kl_target * 2.5 and epoch >= 2:
                break

            local_clip = self.clip_eps if not (do_explore_epoch and epoch == 0) else 0.30

            ratio = torch.exp((logp - lp0).clamp(-20, 20))

            s1 = ratio * adv_eff
            s2 = torch.clamp(ratio, 1.0 - local_clip, 1.0 + local_clip) * adv_eff

            pi_loss = -torch.min(s1, s2).mean()
            kl_pen = self.kl_beta * (logp - lp0).pow(2).mean()

            vR_new = self.critic_r(b_cur)
            qC_new = self.critic_c(b_cur)

            vr_loss = F.mse_loss(vR_new, ret_r_t)

            taus = self.critic_c.taus.to(self.device)
            diff = ret_c_mean_t.unsqueeze(-1) - qC_new

            huber = torch.where(
                diff.abs() < 1.0,
                0.5 * diff ** 2,
                diff.abs() - 0.5,
            )

            vc_loss = (torch.abs(taus - (diff < 0).float()) * huber).mean()

            phase_ce = F.cross_entropy(pl_cur, pl_t, weight=phase_weights)

            attack_logits = self.encoder.attack_logits(b_cur)
            attack_ce = F.cross_entropy(attack_logits, attack_t, weight=attack_weights)

            aux_loss = (0.4 / 0.7) * aux_weight * phase_ce + (0.3 / 0.7) * aux_weight * attack_ce

            last_aux_phase = float(phase_ce.item())
            last_aux_attack = float(attack_ce.item())

            loss = (
                pi_loss
                + 0.5 * (vr_loss + vc_loss)
                - self.ent_coef * ent.mean()
                + aux_loss
                + 0.01 * kl_pen
            )

            self.opt.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(self.opt.param_groups[0]["params"], 0.5)

            self.opt.step()

        if last_kl < self.kl_target / 1.5:
            self.kl_beta = max(0.1, self.kl_beta / 1.5)
        elif last_kl > self.kl_target * 1.5:
            self.kl_beta = min(10.0, self.kl_beta * 1.5)

        if last_ent < 0.8:
            self.ent_coef = min(self.ent_coef_ceil, self.ent_coef * 1.15)
        elif last_ent > 1.8:
            self.ent_coef = max(self.ent_coef_floor, self.ent_coef * 0.92)

        self.ent_coef = max(self.ent_coef_floor, min(self.ent_coef_ceil, self.ent_coef))

        if last_ent < 1.0:
            self.low_ent_streak += 1

            if self.low_ent_streak >= 5:
                self.ent_coef = min(self.ent_coef_ceil, self.ent_coef * 1.3)
                self.low_ent_streak = 0
        else:
            self.low_ent_streak = max(0, self.low_ent_streak - 1)

        if len(self.recent_R) >= 20:
            recent_R_var = float(np.var(list(self.recent_R)[-20:]))
        else:
            recent_R_var = 1e9

        if last_kl < 5e-3 and recent_R_var < 0.5:
            self.dead_zone_streak += 1

            if self.dead_zone_streak >= 3:
                self.ent_coef = min(self.ent_coef_ceil, self.ent_coef * 1.5)
                self.kl_beta = max(0.05, self.kl_beta * 0.5)
                self.request_explore_epoch = True
                self.dead_zone_streak = 0
        else:
            self.dead_zone_streak = max(0, self.dead_zone_streak - 1)

        if last_kl < 1e-4:
            self.kl_zero_streak += 1

            if self.kl_zero_streak >= 20:
                for g in self.opt.param_groups:
                    g["lr"] = max(self.base_lr * 0.3, g["lr"] * 0.9)

                self.kl_zero_streak = 0
        else:
            self.kl_zero_streak = max(0, self.kl_zero_streak - 1)

        return {
            "cvar_b": cvar_b,
            "cvar_a": cvar_a,
            "cvar_j": cvar_joint,
            "lam": lam,
            "lam_raw": lam_raw,
            "kl": last_kl,
            "ent": last_ent,
            "lr": self.opt.param_groups[0]["lr"],
            "beta": self.kl_beta,
            "ec": self.ent_coef,
            "phase_loss": last_aux_phase,
            "attack_loss": last_aux_attack,
            "aux_w": aux_weight,
            "sanity": sanity_action,
            "sat": self.sat_streak,
            "gate_blk": last_gate_block_rate,
            "gate_on": int(gate_active),
            "a1_fb": int(a1_fallback_active),
        }
