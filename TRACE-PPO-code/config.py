"""
TRACE-PPO for CICAPT-IIoT 85-Dim Dataset -- UNSUPERVISED / LABEL-BLIND VERSION
==============================================================================

This version removes ground-truth label usage during training.
Includes Real KL/Entropy Telemetry, GT-aligned AIR/FDR Evaluation, and First-Principles Normalized Checkpoint Logging.

Dataset files expected in the same directory:
  - cicapt_ot_sequence_benign.npy
  - cicapt_ot_sequence_malicious_noleak_context.npy
"""
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

