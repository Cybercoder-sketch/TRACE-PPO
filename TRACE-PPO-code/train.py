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


from config import EXPECTED_OBS_DIM, TOTAL_EPISODES, GLOBAL_SEED, RISK_LOW_TH, RISK_HIGH_TH
from config import SAFE_CKPT_NAME, REWARD_CKPT_NAME, FINAL_CKPT_NAME, FILE_BENIGN, FILE_MALICIOUS
from env import CICAPT_UnsupervisedDataDrivenEnv
from trainer import TRACE_PPO
from utils import setup_logger, get_training_device, logger
# =====================================================
# 7. Training Loop
# =====================================================
def train_data_driven_cicapt_unsupervised(total_eps=TOTAL_EPISODES):
    current_dir = os.path.dirname(os.path.abspath(__file__))

    benign_path = os.path.join(current_dir, FILE_BENIGN)
    mal_path = os.path.join(current_dir, FILE_MALICIOUS)

    device = get_training_device()

    env = CICAPT_UnsupervisedDataDrivenEnv(
        benign_path=benign_path,
        mal_path=mal_path,
        horizon=200,
        V_threshold=0.6,
        V_min_target=0.65,
        idle_bonus=0.03,
        expected_obs_dim=EXPECTED_OBS_DIM,
        risk_low_th=RISK_LOW_TH,
        risk_high_th=RISK_HIGH_TH,
    )

    agent = TRACE_PPO(
        obs_dim=env.obs_dim,
        c_limit_benign=env.c_limit_benign,
        c_limit_attack=env.c_limit_attack,
        attack_rate=0.55,
        gate_low=0.3,
        gate_high=0.7,
        gate_low_fallback=0.12,
        gate_warmup_eps=50,
        pid_start_eps=60,
        pid_ramp_eps=60,
        device=device,
    )

    logger.info(
        f"[CICAPT  UNSUP] Engine Started. "
        f"obs_dim={env.obs_dim}, total_eps={total_eps}, device={device} | "
        f"NO label file is loaded. NO ground-truth labels are used in training. | "
        f"[ATTACK]/[BENIGN] tags in logs mean pseudo high-risk / pseudo low-risk only. | "
        f"V_thr={env.V_threshold}, V_min_tgt={env.V_min_target}, "
        f"idle_bonus={env.idle_bonus} | "
        f"C_lim(low/high/joint)={env.c_limit_benign:.1f}/{env.c_limit_attack:.1f}/"
        f"{agent.c_limit_joint:.2f} | "
        f"w(low/high)={agent.w_b:.2f}/{agent.w_a:.2f} | "
        f"risk_th(low/high)={env.risk_low_th:.2f}/{env.risk_high_th:.2f} | "
        f"lr={agent.base_lr}, lam_max={agent.pid.lam_max}, "
        f"clip_eps={agent.clip_eps}, alpha={agent.alpha} | "
        f"gate=hard(low={agent.actor.gate_low},high={agent.actor.gate_high},"
        f"a1_fb_low={agent.actor.gate_low_fallback}), "
        f"warmup={agent.gate_warmup_eps} | "
        f"pid_start={agent.pid_start_eps}, pid_ramp={agent.pid_ramp_eps} | "
        f"ec_floor={agent.ent_coef_floor}, aux_w=0.25(self-supervised), "
        f"pid(kp={agent.pid.kp},ki={agent.pid.ki},kd={agent.pid.kd},"
        f"rate={agent.pid.rate_limit}), "
        f"cost_buf={agent.cost_buf_mal.maxlen}+recent={agent.cost_recent_mal.maxlen}, "
        f"safe_ckpt={agent.safe_ckpt_path}"
    )

    # [后台暗账记录]: 使用真实的 Ground-Truth 统计
    recent_ep_attack_steps_gt = deque(maxlen=30)
    recent_ep_intercepts_gt = deque(maxlen=30)

    best_fallback_reward = -1e9
    reward_ckpt_path = os.path.join(os.getcwd(), REWARD_CKPT_NAME)

    for ep in range(total_eps):
        buf = {
            k: []
            for k in [
                "obs",
                "beliefs",
                "phase_logits",
                "attack_prob",
                "a_type",
                "a_mu",
                "logp_old",
                "rewards",
                "costs",
                "dones",
            ]
        }

        pseudo_phase_labels = []

        o, phase = env.reset()
        b = agent.encoder.init_belief(1, agent.device)

        done = False
        info = {}

        gate_active_now = agent.episode_count >= agent.gate_warmup_eps
        a1_fallback_now = agent.risk_fallback_active()

        buf["a1_fallback_active"] = bool(a1_fallback_now)

        a0_count = 0
        a1_count = 0
        a2_count = 0
        step_count = 0

        while not done:
            o_t = torch.as_tensor(o, dtype=torch.float32, device=agent.device).unsqueeze(0)

            with torch.no_grad():
                b = agent.encoder(o_t, b)
                pl = agent.encoder.phase_logits(b)
                ap = agent.encoder.attack_prob(b)

                a_type, a_mu, logp, _ = agent.actor.act(
                    b,
                    pl,
                    ap,
                    gate_active=gate_active_now,
                    a1_fallback_active=a1_fallback_now,
                )

            o2, r, c, done, phase2, info = env.step(a_type.item(), a_mu.item())

            buf["obs"].append(o.copy())
            buf["beliefs"].append(b.detach().cpu().numpy()[0])
            buf["phase_logits"].append(pl.detach().cpu().numpy()[0])
            buf["attack_prob"].append(float(ap.item()))
            buf["a_type"].append(a_type.item())
            buf["a_mu"].append(a_mu.item())
            buf["logp_old"].append(logp.item())
            buf["rewards"].append(r)
            buf["costs"].append(c)
            buf["dones"].append(float(done))

            pseudo_phase_labels.append(int(np.clip(phase, 0, 2)))

            step_count += 1

            if a_type.item() == 0:
                a0_count += 1
            elif a_type.item() == 1:
                a1_count += 1
            else:
                a2_count += 1

            o = o2
            phase = phase2

        R_ep = float(np.sum(buf["rewards"]))
        C_ep = float(np.sum(buf["costs"]))

        V_drop_ep = float(info.get("V_drop_ep", 0.0))
        V_min_ep = float(info.get("V_min_ep", 1.0))

        ep_avg_risk = float(info.get("ep_avg_risk", 0.0))
        pseudo_high_episode = bool(info.get("pseudo_high_episode", False))

        # 接收真实的 Ground-Truth 评价指标
        recent_ep_attack_steps_gt.append(int(info.get("ep_attack_steps_gt", 0)))
        recent_ep_intercepts_gt.append(int(info.get("ep_intercepts_gt", 0)))
        is_malicious_gt = bool(info.get("is_malicious_gt", False))

        a0_ratio = a0_count / max(1, step_count)
        a1_ratio = a1_count / max(1, step_count)
        a2_ratio = a2_count / max(1, step_count)

        agent.recent_R.append(R_ep)
        agent.recent_C.append(C_ep)

        # 这里的缓冲用于算 CVaR，依然使用基于无监督异常分的 pseudo 标签进行物理分层
        if pseudo_high_episode:
            agent.recent_C_a.append(C_ep)
            agent.recent_Vmin_a.append(V_min_ep)
            agent.cost_buf_mal.append(V_drop_ep)
            agent.cost_recent_mal.append(V_drop_ep)
            agent.recent_a2_a.append(a2_ratio)
        else:
            agent.recent_C_b.append(C_ep)
            agent.recent_Vmin_b.append(V_min_ep)
            agent.cost_buf_benign.append(V_drop_ep)
            agent.cost_recent_benign.append(V_drop_ep)
            agent.recent_a2_b.append(a2_ratio)

        try:
            stats = agent.update(buf, pseudo_phase_labels)
        except Exception as e:
            logger.info(f"[Ep {ep}] Update Skip: {repr(e)}")
            agent.episode_count += 1
            continue

        agent.episode_count += 1
        rollback_fired = False

        if len(agent.recent_R) >= 30:
            avg_R = float(np.mean(agent.recent_R))

            avg_Vmin_b = float(np.mean(agent.recent_Vmin_b)) if agent.recent_Vmin_b else 1.0
            avg_Vmin_a = float(np.mean(agent.recent_Vmin_a)) if agent.recent_Vmin_a else 1.0

            avg_a2_b = float(np.mean(agent.recent_a2_b)) if agent.recent_a2_b else 0.0
            avg_a2_a = float(np.mean(agent.recent_a2_a)) if agent.recent_a2_a else 0.0

            sla_b = avg_Vmin_b >= 0.5 * env.V_min_target
            sla_a = avg_Vmin_a >= 0.5 * env.V_min_target

            # 内部保存状态逻辑 (不改变原有的 rollback 机制)
            if avg_R > agent.best_R_avg and sla_b and sla_a:
                agent.best_R_avg = avg_R
                agent._save_snapshot()

            if agent.best_R_avg > 0.0:
                agent.best_R_avg *= 0.9985

            # 第一阶段：安全门控函数 (Safety Gating)
            safe_ok = (
                stats["cvar_j"] < agent.c_limit_joint
                and stats["lam"] < 5.0
                and avg_Vmin_a >= 0.68
                and avg_a2_b <= 0.02
            )

            # [真实评价体系]: 严格遵循论文中的 AIR 定义 (TP / TP + FN)
            total_atk_gt = sum(recent_ep_attack_steps_gt)
            air_percent = (sum(recent_ep_intercepts_gt) / total_atk_gt * 100.0) if total_atk_gt > 0 else 0.0

            # --- [新增/修改] 第二阶段：对合格模型做归一化综合评分 (Normalized Satisfaction Scoring) ---
            # 1. 设定边界参数 (依据物理业务极限设定)
            R_min, R_target = 0.0, 100.0               # 奖励的经验下限与理论上限
            A_min, A_target = 0.0, 100.0               # AIR 真实防御率 (%)
            V_floor, V_target = 0.68, 1.0              # 生存度从 SLA 底线到完美无损
            C_target, C_max = 0.0, agent.c_limit_joint # CVaR 从 0 风险到红线极限
            F_target, F_max = 0.0, 0.02                # FDR(误杀率) 从 0 到容忍极限

            # 2. 计算各维度的归一化满意度 (clip 到 0~1)
            # 添加 1e-8 防止除以 0
            s_R = np.clip((avg_R - R_min) / (R_target - R_min + 1e-8), 0.0, 1.0)
            s_A = np.clip((air_percent - A_min) / (A_target - A_min + 1e-8), 0.0, 1.0)
            s_V = np.clip((avg_Vmin_a - V_floor) / (V_target - V_floor + 1e-8), 0.0, 1.0)
            s_C = np.clip((C_max - stats["cvar_j"]) / (C_max - C_target + 1e-8), 0.0, 1.0)
            s_F = np.clip((F_max - avg_a2_b) / (F_max - F_target + 1e-8), 0.0, 1.0)

            # 3. 基于 APT 防御第一性原理的加权组合
            # 权重: AIR(35%), Vmin(30%), CVaR(20%), FDR(10%), avgR(5%)
            safe_score = 0.05 * s_R + 0.35 * s_A + 0.30 * s_V + 0.10 * s_F + 0.20 * s_C
            # -------------------------------------------------------------------------------------

            a_mix = f"{a0_ratio:.2f}/{a1_ratio:.2f}/{a2_ratio:.2f}"

            # 1. Best Safe 检查点保存与面板日志
            if safe_ok and safe_score > agent.best_safe_score:
                agent.best_safe_score = safe_score
                agent.best_safe_ep = ep
                torch.save(agent._state_dict_for_ckpt(), agent.safe_ckpt_path)
                
                logger.info(
                    f"\n=======================================================\n"
                    f"[Ep {ep}] *** BEST-SAFE CHECKPOINT SAVED ***\n"
                    f"   -> Path: {agent.safe_ckpt_path}\n"
                    f"   -> Score: {safe_score:.4f} (Max: 1.0) | avgR: {avg_R:.2f}\n"
                    f"   -> Vmin(b/a): {avg_Vmin_b:.2f}/{avg_Vmin_a:.2f} | CVaR_j: {stats['cvar_j']:.2f} (Lim:{agent.c_limit_joint:.2f})\n"
                    f"   -> AIR (GT): {air_percent:.1f}% | benign_a2 (FDR): {avg_a2_b:.3f} | a_mix: {a_mix}\n"
                    f"   -> KL: {stats['kl']:.4f} | ent: {stats['ent']:.3f} | lam: {stats['lam']:.2f} | risk: {ep_avg_risk:.3f}\n"
                    f"=======================================================\n"
                )

            # 2. [新增] Best Reward 回退检查点保存与日志
            if avg_R > best_fallback_reward and ep > 20:
                best_fallback_reward = avg_R
                torch.save(agent._state_dict_for_ckpt(), reward_ckpt_path)

                logger.info(
                    f"\n-------------------------------------------------------\n"
                    f"[Ep {ep}] >>> BEST-REWARD CHECKPOINT SAVED (Fallback) <<<\n"
                    f"   -> Path: {reward_ckpt_path}\n"
                    f"   -> avgR: {avg_R:.2f} | Score: {safe_score:.4f}\n"
                    f"   -> Vmin(b/a): {avg_Vmin_b:.2f}/{avg_Vmin_a:.2f} | CVaR_j: {stats['cvar_j']:.2f}\n"
                    f"   -> AIR (GT): {air_percent:.1f}% | benign_a2: {avg_a2_b:.3f} | a_mix: {a_mix}\n"
                    f"   -> KL: {stats['kl']:.4f} | ent: {stats['ent']:.3f} | lam: {stats['lam']:.2f}\n"
                    f"   -> SLA STATUS: {'Satisfied (Safe)' if safe_ok else 'VIOLATED! Kept for comparative analysis.'}\n"
                    f"-------------------------------------------------------\n"
                )

            cooled = (ep - agent.last_rollback_ep) >= agent.rollback_cooldown
            big_drop = (agent.best_R_avg > 5.0) and (
                avg_R < agent.best_R_avg - agent.underperf_threshold
            )

            if big_drop and cooled:
                agent.underperf_streak += 1

                if agent.underperf_streak >= 2:
                    logger.info(
                        f"[Ep {ep}] *** Rollback to best snapshot "
                        f"(avgR {avg_R:.2f} << best {agent.best_R_avg:.2f}) ***"
                    )

                    agent._load_snapshot()
                    agent.last_rollback_ep = ep
                    agent.underperf_streak = 0
                    agent.best_R_avg = max(avg_R, agent.best_R_avg - 5.0)

                    agent.pid.clear(hard=True)
                    agent.sat_streak = 0
                    agent.clear_cost_buffers()

                    rollback_fired = True
            else:
                agent.underperf_streak = max(0, agent.underperf_streak - 1)

        if ep % 10 == 0 or rollback_fired:
            # 严格根据实际 Ground-Truth 判定记录显示标签，避免误导
            env_type = "[ATTACK]" if is_malicious_gt else "[BENIGN]"

            avg_R = float(np.mean(agent.recent_R)) if agent.recent_R else 0.0
            avg_C_b = float(np.mean(agent.recent_C_b)) if agent.recent_C_b else 0.0
            avg_C_a = float(np.mean(agent.recent_C_a)) if agent.recent_C_a else 0.0

            avg_Vmin_b = float(np.mean(agent.recent_Vmin_b)) if agent.recent_Vmin_b else 1.0
            avg_Vmin_a = float(np.mean(agent.recent_Vmin_a)) if agent.recent_Vmin_a else 1.0

            avg_a2_b = float(np.mean(agent.recent_a2_b)) if agent.recent_a2_b else 0.0

            a_mix = f"{a0_ratio:.2f}/{a1_ratio:.2f}/{a2_ratio:.2f}"
            sanity_tag = "" if stats["sanity"] == "none" else f" sanity={stats['sanity']}"
            
            # 使用真实 GT 计算全局打印日志
            total_atk_gt = sum(recent_ep_attack_steps_gt)
            air_percent = (sum(recent_ep_intercepts_gt) / total_atk_gt * 100.0) if total_atk_gt > 0 else 0.0

            logger.info(
                f"Ep {ep:4d} {env_type:8s} | "
                f"R={R_ep:6.2f} avgR={avg_R:6.2f} | "
                f"Vmin(b/a)={avg_Vmin_b:.2f}/{avg_Vmin_a:.2f} "
                f"Vdrop={V_drop_ep:5.1f} | "
                f"Cb={avg_C_b:5.1f} Ca={avg_C_a:6.1f} | "
                f"CVaR(b/a/J)="
                f"{stats['cvar_b']:4.1f}/{stats['cvar_a']:5.1f}/{stats['cvar_j']:5.2f}"
                f"(lim={agent.c_limit_joint:.2f}) | "
                f"λ={stats['lam']:5.2f}/{agent.pid.lam_max:.0f} "
                f"raw={stats['lam_raw']:5.2f} sat={stats['sat']:2d} | "
                f"a012={a_mix} bA2avg={avg_a2_b:.3f} | "
                f"AIR={air_percent:.1f}% "
                f"gate(on={stats['gate_on']},blk={stats['gate_blk']:.2f},fb={stats['a1_fb']}) | "
                f"KL={stats['kl']:.4f} ent={stats['ent']:.3f} | "
                f"phs={stats['phase_loss']:.3f} atk={stats['attack_loss']:.3f} | "
                f"β={stats['beta']:.2f} ec={stats['ec']:.3f} aw={stats['aux_w']:.2f} | "
                f"risk={ep_avg_risk:.3f} | "
                f"lr={stats['lr']:.2e} bestSafeEp={agent.best_safe_ep}{sanity_tag}"
            )

    final_path = os.path.join(os.getcwd(), FINAL_CKPT_NAME)

    torch.save(agent._state_dict_for_ckpt(), final_path)

    logger.info(f"[Done] Final checkpoint saved: {final_path}")

    return agent


# =====================================================
# 8. Main
# =====================================================
if __name__ == "__main__":
    torch.manual_seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(GLOBAL_SEED)
        torch.backends.cudnn.benchmark = True

    train_data_driven_cicapt_unsupervised(total_eps=TOTAL_EPISODES)