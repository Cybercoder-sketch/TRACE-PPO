# TRACE-PPO for CICAPT-IIoT

This repository contains the official implementation of **TRACE-PPO** (Unsupervised/Label-Blind Version), a Safe Reinforcement Learning framework tailored for Advanced Persistent Threat (APT) detection and dynamic defense in Industrial Internet of Things (IIoT) environments.

## Repository Structure

\TRACE-PPO-code/
├── config.py             # Global configurations, hyperparameters, and file paths
├── models.py             # Neural network architectures (BeliefEncoder, HybridActor, Critics)
├── env.py                # Data-driven unsupervised environment wrapper and reward metrics
├── utils.py              # Helper functions, metric calculations, and PID lagrangian logic
├── trainer.py            # Core TRACE-PPO trainer class
├── train.py              # Main entry point for starting the training loop
├── dataset_tools/        # Scripts for processing the CICAPT-IIoT dataset
│   ├── extract_malicious.py   # Extracts malicious sequences to .npy
│   ├── extract_benign.py      # Extracts benign sequences to .npy
│   └── patch_label_leak.py    # Patch for addressing label leakage in malicious extraction
└── README.md             # This file
\
## Dataset

This code evaluates the TRACE-PPO algorithm using the **CICAPT-IIoT** dataset, developed by the **Canadian Institute for Cybersecurity (CIC)** at the **University of New Brunswick (UNB)**. The dataset provides a comprehensive, provenance-based foundation for APT detection in IIoT, combining network traffic and provenance logs.

*   **Official Dataset Link**: [UNB CIC Datasets (CICAPT-IIoT 2024)](https://www.unb.ca/cic/datasets/iiot-dataset-2024.html)

### Data Preprocessing

Before training, the raw CSV dataset needs to be ingested and converted into window-level .npy sequences:

1. **Download the Data**: Download the official CICAPT-IIoT network and provenance CSV files.
2. **Extract Malicious Sequences**:
   \\ash
   python dataset_tools/extract_malicious.py
   \3. **Apply Label Leakage Patch**:
   \\ash
   python dataset_tools/patch_label_leak.py
   \4. **Extract Benign Sequences**:
   \\ash
   python dataset_tools/extract_benign.py
   \
After preprocessing, make sure the output \.npy\ files (\cicapt_ot_sequence_benign.npy\ and \cicapt_ot_sequence_malicious_noleak.npy\) are placed in the root directory (or update the paths in \config.py\).

## Training

To train the TRACE-PPO agent, simply run the \	rain.py\ script. The script is configured to run without ground-truth label usage during training (label-blind), relying on unsupervised anomaly scores for its reward formulation.

\\ash
# Run training with default seed
python train.py

# Run training with a specific seed
python train.py 42
\
### Logging and Checkpoints

*   **Logs**: Training logs will be generated in the root directory in the format \cicapt_training_seed{SEED}_unsup_{TIMESTAMP}.log\.
*   **Checkpoints**: The trainer will automatically save models based on physical SLA constraints:
    *   \TRACE-PPO_seed{SEED}_best_safe.pt\: Best model satisfying survivability (SLA) constraints.
    *   \TRACE-PPO_seed{SEED}_best_reward.pt\: Model with highest reward (fallback).
    *   \cicapt_Trace_unsup_final.pt\: Final model at the end of the training epochs.
