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


from utils import load_feature_matrix, SurvivabilityMetric, logger
# =====================================================
# 0. Global Config
# =====================================================
EXPECTED_OBS_DIM = 85
TOTAL_EPISODES = 1890
GLOBAL_SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42

# --- 请将你代码中原来的这几行写死的常量，替换为带 seed{GLOBAL_SEED} 的动态命名 ---
# （注意：不同模型的命名前缀不同，请保留各自的前缀，只把 seed 拼进去）
SAFE_CKPT_NAME = f"TRACE-PPO_seed{GLOBAL_SEED}_best_safe.pt"
REWARD_CKPT_NAME = f"TRACE-PPO_seed{GLOBAL_SEED}_best_reward.pt"
FILE_BENIGN = "cicapt_ot_sequence_benign.npy"
FILE_MALICIOUS = "cicapt_ot_sequence_malicious_noleak.npy"

# SAFE_CKPT_NAME = "cicapt_Trace_best_safe.pt"
# REWARD_CKPT_NAME = "cicapt_Trace_unsup_best_reward.pt"
FINAL_CKPT_NAME = "cicapt_Trace_unsup_final.pt"

SHUFFLE_UNLABELED_POOL = True
RISK_LOW_TH = 0.35
RISK_HIGH_TH = 0.65

# =====================================================
# 5. Unsupervised CICAPT-IIoT Data-Driven Env
# =====================================================
class CICAPT_UnsupervisedDataDrivenEnv:
    def __init__(
        self,
        benign_path,
        mal_path,
        horizon=200,
        V_threshold=0.6,
        V_min_target=0.65,
        idle_bonus=0.03,
        expected_obs_dim=85,
        risk_low_th=RISK_LOW_TH,
        risk_high_th=RISK_HIGH_TH,
    ):
        self.horizon = horizon

        benign_data = load_feature_matrix(
            benign_path,
            name="CICAPT file-A",
            expected_dim=expected_obs_dim,
        )

        mal_data = load_feature_matrix(
            mal_path,
            name="CICAPT file-B",
            expected_dim=expected_obs_dim,
        )

        self.dataset = np.concatenate([benign_data, mal_data], axis=0).astype(np.float32)

        # [后台暗账]: 创建对应的 Ground-Truth 标签，仅用于评估日志，绝不参与训练
        benign_labels = np.zeros(len(benign_data), dtype=np.int32)
        mal_labels = np.ones(len(mal_data), dtype=np.int32)
        all_labels = np.concatenate([benign_labels, mal_labels], axis=0)

        if SHUFFLE_UNLABELED_POOL:
            perm = np.random.permutation(len(self.dataset))
            self.dataset = self.dataset[perm]
            self.dataset_labels = all_labels[perm]
        else:
            self.dataset_labels = all_labels

        if len(self.dataset) <= self.horizon + 1:
            raise ValueError(
                f"[Env] Unlabeled data too short for horizon={self.horizon}: "
                f"data_len={len(self.dataset)}"
            )

        self.obs_dim = int(self.dataset.shape[1])

        if self.obs_dim != expected_obs_dim:
            raise ValueError(
                f"[Env] obs_dim mismatch: expected={expected_obs_dim}, got={self.obs_dim}"
            )

        logger.info(
            f"[Env] Unsupervised pool loaded. "
            f"Unlabeled data: {self.dataset.shape}, obs_dim={self.obs_dim}. "
            f"Ground-Truth labels preserved for evaluation logging ONLY."
        )

        self.risk_low_th = risk_low_th
        self.risk_high_th = risk_high_th
        self._fit_unsupervised_risk_model(self.dataset)

        self.base_w = [0.2, 0.9]
        self.metric = SurvivabilityMetric(self.base_w, beta=0.587, rebound=0.05)

        self.V_threshold = V_threshold
        self.V_min_target = V_min_target
        self.idle_bonus = idle_bonus

        self.c_limit_benign = 5.0
        self.c_limit_attack = 20.0

        self.reset()

    def _fit_unsupervised_risk_model(self, data):
        med = np.median(data, axis=0)
        mad = np.median(np.abs(data - med), axis=0)
        scale = 1.4826 * mad + 1e-6

        z = np.clip((data - med) / scale, -10.0, 10.0)
        norm = np.sqrt(np.mean(z ** 2, axis=1))

        q50 = float(np.quantile(norm, 0.50))
        q95 = float(np.quantile(norm, 0.95))

        if q95 <= q50 + 1e-6:
            q95 = q50 + 1.0

        self.risk_med = med.astype(np.float32)
        self.risk_scale = scale.astype(np.float32)
        self.risk_q50 = q50
        self.risk_q95 = q95

    def _risk_score_obs(self, obs):
        z = np.clip((obs.astype(np.float32) - self.risk_med) / self.risk_scale, -10.0, 10.0)
        norm = float(np.sqrt(np.mean(z ** 2)))
        score = (norm - self.risk_q50) / (self.risk_q95 - self.risk_q50 + 1e-8)
        return float(np.clip(score, 0.0, 1.0))

    def _risk_phase(self, risk_score):
        if risk_score >= self.risk_high_th:
            return 2
        if risk_score >= self.risk_low_th:
            return 1
        return 0

    def _get_obs(self):
        idx = min(self.start_idx + self.t, len(self.dataset) - 1)
        return self.dataset[idx].astype(np.float32)

    def _current_risk(self):
        return self._risk_score_obs(self._get_obs())

    def _current_pseudo_phase(self):
        return self._risk_phase(self._current_risk())

    def reset(self):
        self.t = 0

        max_start = max(0, len(self.dataset) - self.horizon - 1)
        self.start_idx = np.random.randint(0, max_start + 1)

        self.s_phy = np.array([1.0, 1.0])
        self.metric.tau = np.zeros(2)

        self.V_prev = self.metric.survivability(self.s_phy)
        self.V_min_ep = self.V_prev
        self.episode_cost_sum = 0.0

        self.ep_high_risk_steps = 0
        self.ep_high_risk_responses = 0
        self.ep_low_risk_steps = 0
        self.ep_low_risk_a2 = 0
        self.ep_risk_sum = 0.0

        # [后台暗账统计]: 恢复真实的 Ground-Truth 判定，严格对齐论文手稿
        self.is_malicious_gt = bool(self.dataset_labels[self.start_idx] == 1)
        if self.is_malicious_gt:
            self.attack_phase_gt = int(np.random.choice([1, 1, 2]))
        else:
            self.attack_phase_gt = 0

        self.ep_attack_steps_gt = 0
        self.ep_intercepts_gt = 0

        # Agent 看到的仍然是基于特征算出的 Pseudo Phase
        phase_label = self._current_pseudo_phase()

        return self._get_obs(), phase_label

    def step(self, a_type, a_mu):
        # 1. 所有的状态转移、惩罚、奖励，完全依赖无监督的 risk
        risk = self._current_risk()
        phase = self._risk_phase(risk)

        self.ep_risk_sum += risk

        if risk >= self.risk_high_th:
            self.ep_high_risk_steps += 1
        if risk < self.risk_low_th:
            self.ep_low_risk_steps += 1

        if a_type == 0:
            s_c, s_b, mig = 1.0, 1.0, [0, 0]
            mitigation = 0.0
            action_cost = 0.0
        elif a_type == 1:
            s_c, s_b, mig = 1.0, 1.0, [0, a_mu]
            mitigation = 0.45 * a_mu
            action_cost = 0.05 * a_mu
        else:
            s_c, s_b, mig = 1.0 - 0.5 * a_mu, 1.0 - a_mu, [0, 0]
            mitigation = 0.75 * a_mu
            action_cost = 0.15 + 0.30 * a_mu

        if risk >= self.risk_high_th and a_type > 0 and mitigation >= 0.25 * risk:
            self.ep_high_risk_responses += 1

        if risk < self.risk_low_th and a_type == 2:
            self.ep_low_risk_a2 += 1

        self.metric.update_tau(mig)
        self.s_phy = np.array([s_c, s_b])

        V_now = self.metric.survivability(self.s_phy)
        self.V_min_ep = min(self.V_min_ep, V_now)
        self.V_prev = V_now

        v_short = max(0.0, self.V_min_target - V_now)
        residual_risk = max(0.0, risk - mitigation)

        cost = 10.0 * v_short
        cost += 3.0 * max(0.0, risk - self.risk_high_th) * residual_risk
        cost += 1.5 * action_cost

        if risk < self.risk_low_th and a_type == 2:
            cost += 2.0 * a_mu

        reward = 0.02
        reward += 0.50 * V_now
        reward -= 0.15 * v_short
        reward -= action_cost

        if risk >= self.risk_high_th:
            reward += 1.2 * min(risk, mitigation)
            reward -= 0.45 * residual_risk

            if a_type == 0:
                reward -= 0.10
        elif risk >= self.risk_low_th:
            reward += 0.45 * min(risk, mitigation)
            reward -= 0.12 * residual_risk
        else:
            if a_type == 0:
                reward += self.idle_bonus
            else:
                reward -= 0.20 * a_mu

            if a_type == 2:
                reward -= 0.35 * a_mu

        self.episode_cost_sum += cost

        # 2. [后台暗账统计]: 仅仅记录 Ground Truth 的 TP 和 FN，与 Agent 梯度彻底隔离
        intercept_prob_gt = {0: 0.05, 1: 0.4 + 0.3 * a_mu, 2: 0.8 + 0.2 * a_mu}[a_type]
        
        if self.is_malicious_gt:
            if self.attack_phase_gt >= 1:
                self.ep_attack_steps_gt += 1
            intercepted_gt = (np.random.rand() < intercept_prob_gt) and (self.attack_phase_gt >= 1)

            if intercepted_gt:
                self.ep_intercepts_gt += 1
                self.attack_phase_gt = max(0, self.attack_phase_gt - 1)
            else:
                if np.random.rand() < 0.2:
                    self.attack_phase_gt = min(2, self.attack_phase_gt + 1)
        else:
            self.attack_phase_gt = 0

        self.t += 1
        done = (self.t >= self.horizon) or (s_b <= 0.01)

        info = {}

        if done:
            v_short_ep = 100.0 * max(0.0, self.V_min_target - self.V_min_ep)
            avg_risk = self.ep_risk_sum / max(1, self.t)

            risk_ep = v_short_ep + 0.05 * self.episode_cost_sum
            risk_ep += 8.0 * max(0.0, avg_risk - self.risk_high_th)

            info["V_drop_ep"] = float(risk_ep)
            info["V_short_ep"] = float(v_short_ep)
            info["V_min_ep"] = float(self.V_min_ep)

            info["ep_avg_risk"] = float(avg_risk)

            info["pseudo_high_episode"] = bool(
                avg_risk >= 0.50 or self.ep_high_risk_steps >= 0.25 * self.horizon
            )
            
            # 传递给日志系统的真实物理评价指标
            info["ep_attack_steps_gt"] = self.ep_attack_steps_gt
            info["ep_intercepts_gt"] = self.ep_intercepts_gt
            info["is_malicious_gt"] = self.is_malicious_gt

        next_phase_label = self._current_pseudo_phase()

        return self._get_obs(), reward, cost, done, next_phase_label, info
