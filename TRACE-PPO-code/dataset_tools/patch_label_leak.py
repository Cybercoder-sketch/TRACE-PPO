import numpy as np
import pandas as pd
from pathlib import Path
import shutil

# ============================================================
# Config
# ============================================================

X_NPY = Path("cicapt_ot_sequence_malicious.npy")
FEATURE_COLS_CSV = Path("cicapt_feature_columns.csv")

DROP_FEATURES = [
    "prov_num_mean__label",
    "prov_num_mean__subLabel",
]

OUT_X_NPY = X_NPY.with_name(X_NPY.stem + "_noleak.npy")
OUT_FEATURE_COLS_CSV = FEATURE_COLS_CSV.with_name(FEATURE_COLS_CSV.stem + "_noleak.csv")

# 如果你确认要原地替换，可以改成 True。
# 默认 False 更安全。
INPLACE_REPLACE = False


# ============================================================
# Helper: robustly load feature columns
# ============================================================

def load_feature_columns(csv_path, expected_dim):
    """
    兼容两种 feature_columns.csv:
    1) 有表头: feature
    2) 无表头: 第一行就是特征名
    """
    df_header = pd.read_csv(csv_path)
    cols_header = df_header.iloc[:, 0].astype(str).tolist()

    df_no_header = pd.read_csv(csv_path, header=None)
    cols_no_header = df_no_header.iloc[:, 0].astype(str).tolist()

    if len(cols_header) == expected_dim:
        return cols_header, True, df_header.columns[0]

    if len(cols_no_header) == expected_dim:
        return cols_no_header, False, None

    raise ValueError(
        f"Feature column number does not match X dimension. "
        f"X dim={expected_dim}, "
        f"with header={len(cols_header)}, "
        f"without header={len(cols_no_header)}"
    )


def save_feature_columns(cols, out_path, has_header=True, header_name="feature"):
    if has_header:
        pd.DataFrame({header_name: cols}).to_csv(out_path, index=False)
    else:
        pd.DataFrame(cols).to_csv(out_path, index=False, header=False)


# ============================================================
# Main
# ============================================================

print("=" * 80)
print("Loading X npy...")

X = np.load(X_NPY, mmap_mode="r")

if X.ndim != 2:
    raise ValueError(f"Expected 2D feature matrix, but got shape {X.shape}")

n_samples, n_features = X.shape
print(f"Original X shape: {X.shape}")

feature_cols, has_header, header_name = load_feature_columns(
    FEATURE_COLS_CSV,
    expected_dim=n_features
)

print(f"Original feature columns: {len(feature_cols)}")

# 检查待删除列是否存在
missing_drop_cols = [c for c in DROP_FEATURES if c not in feature_cols]
if missing_drop_cols:
    raise ValueError(f"Drop columns not found in feature list: {missing_drop_cols}")

drop_indices = [feature_cols.index(c) for c in DROP_FEATURES]
keep_indices = [i for i in range(n_features) if i not in drop_indices]

new_feature_cols = [feature_cols[i] for i in keep_indices]

print("\nDropping leakage features:")
for c, idx in zip(DROP_FEATURES, drop_indices):
    print(f"  - index={idx:4d}, feature={c}")

print(f"\nNew feature dimension: {len(new_feature_cols)}")

# 删除列
X_new = np.asarray(X[:, keep_indices], dtype=X.dtype)

print(f"New X shape: {X_new.shape}")

# 保存新文件
np.save(OUT_X_NPY, X_new)
save_feature_columns(
    new_feature_cols,
    OUT_FEATURE_COLS_CSV,
    has_header=has_header,
    header_name=header_name if header_name is not None else "feature"
)

print("\nSaved:")
print(f"  X: {OUT_X_NPY}")
print(f"  feature columns: {OUT_FEATURE_COLS_CSV}")

# 校验
X_check = np.load(OUT_X_NPY, mmap_mode="r")
cols_check, _, _ = load_feature_columns(
    OUT_FEATURE_COLS_CSV,
    expected_dim=X_check.shape[1]
)

assert X_check.shape[1] == len(cols_check)
assert "prov_num_mean__label" not in cols_check
assert "prov_num_mean__subLabel" not in cols_check

print("\nValidation passed.")
print(f"Final X shape: {X_check.shape}")

# 可选：原地替换
if INPLACE_REPLACE:
    backup_x = X_NPY.with_suffix(X_NPY.suffix + ".bak")
    backup_cols = FEATURE_COLS_CSV.with_suffix(FEATURE_COLS_CSV.suffix + ".bak")

    shutil.copy2(X_NPY, backup_x)
    shutil.copy2(FEATURE_COLS_CSV, backup_cols)

    shutil.copy2(OUT_X_NPY, X_NPY)
    shutil.copy2(OUT_FEATURE_COLS_CSV, FEATURE_COLS_CSV)

    print("\nINPLACE_REPLACE=True")
    print(f"Backed up original X to: {backup_x}")
    print(f"Backed up original feature columns to: {backup_cols}")
    print("Original files have been replaced.")

print("=" * 80)