# -*- coding: utf-8 -*-
"""
CICAPT-IIoT Phase-1 Pure Benign Data Ingestion Pipeline V2.0

Target:
    1. Extract pure benign Phase-1 network/provenance data.
    2. Use the exact same 5-second window aggregation logic as malicious extraction.
    3. Align feature columns strictly with cicapt_feature_columns.csv from malicious extraction.
    4. Generate fixed-size benign .npy with 1890 rows and 85 dimensions.

Input:
    phase1_NetworkData.csv
    Phase1_Provenance.csv
    cicapt_feature_columns.csv

Output:
    cicapt_ot_sequence_benign.npy
    cicapt_ot_sequence_benign_raw.npy
    cicapt_ot_labels_benign_zero.npy
    cicapt_benign_feature_columns.csv
    cicapt_benign_label_columns.csv
    cicapt_benign_window_index.csv
    cicapt_benign_extraction_summary.json
"""

import os
import re
import json
import logging
import warnings
from collections import defaultdict
from functools import reduce

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# =====================================================
# 1. 基础配置
# =====================================================
TIME_WINDOW_SEC = 5.0

# 为避免内存溢出，这里不要设置太大
CHUNK_SIZE = 300_000

FILE_NETWORK = "phase1_NetworkData.csv"
FILE_PROVENANCE = "Phase1_Provenance.csv"

# 必须来自恶意抽取代码的输出，用于保证 85 维完全对应
FILE_REF_FEATURE_COLUMNS = "cicapt_feature_columns_noleak.csv"

# 可选：用于校验良性与恶意特征维度是否一致 
FILE_ATTACK_FEATURES = "cicapt_ot_sequence_malicious_noleak.npy"

OUTPUT_FEATURES = "cicapt_ot_sequence_benign.npy"
OUTPUT_FEATURES_RAW = "cicapt_ot_sequence_benign_raw.npy"
OUTPUT_LABELS = "cicapt_ot_labels_benign_zero.npy"

OUTPUT_FEATURE_COLUMNS = "cicapt_benign_feature_columns.csv"
OUTPUT_LABEL_COLUMNS = "cicapt_benign_label_columns.csv"
OUTPUT_WINDOW_INDEX = "cicapt_benign_window_index.csv"
OUTPUT_SUMMARY = "cicapt_benign_extraction_summary.json"

TARGET_ROWS = 1890
EXPECTED_FEATURE_DIM = 85

# head: 取时间顺序最早的 1890 个良性窗口，适合保持时序连续性
# uniform: 在所有良性窗口中均匀抽取 1890 个窗口，适合覆盖更完整背景流量
SAMPLE_MODE = "head"

# Phase-1 是纯良性，这里仍保留 label 安全过滤
ENABLE_BENIGN_FILTER = True

# 良性标签不是训练必须项，但保存全 0 标签可以兼容恶意抽取的 4 维 label 结构
SAVE_ZERO_LABELS = True

# 与恶意抽取代码保持一致，默认不做 z-score
APPLY_ZSCORE = False
EPS = 1e-8

# 不建议在良性抽取中强行生成完整连续时间轴，否则容易产生大量空窗口
CREATE_CONTINUOUS_TIMELINE = False
MAX_CONTINUOUS_WINDOWS = 2_000_000

NETWORK_TS_CANDIDATES = [
    "ts", "timestamp", "time", "Time", "datetime", "date_time"
]

PROVENANCE_TIME_CANDIDATES = [
    "time", "seen time", "seen_time", "timestamp", "ts", "event_time", "datetime"
]

PROV_TEXT_HINTS = [
    "type", "event", "operation", "action", "relation", "predicate",
    "object", "subject", "process", "file", "path", "socket",
    "cmd", "command", "name"
]

KEYWORD_FEATURES = {
    "prov_file_event_count": r"file|path|read|write|open|close|delete|unlink|rename",
    "prov_process_event_count": r"process|proc|exec|fork|spawn|pid|cmd|command",
    "prov_network_event_count": r"socket|connect|send|recv|receive|accept|dns|ip|port|network",
    "prov_read_event_count": r"read|recv|receive",
    "prov_write_event_count": r"write|send|create|modify|append",
    "prov_exec_event_count": r"exec|execute|fork|spawn|process|cmd|command",
    "prov_delete_event_count": r"delete|unlink|remove|rm",
    "prov_connect_event_count": r"connect|socket|accept|dns|port|ip"
}

PROV_NUM_PREFIX = "prov_num_mean__"
PROV_UNIQUE_PREFIX = "prov_unique__"


# =====================================================
# 2. 日志初始化
# =====================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("CICAPT_Phase1_Benign_Extractor")


# =====================================================
# 3. 通用工具函数
# =====================================================
def safe_name(name: str) -> str:
    name = str(name)
    name = re.sub(r"[^0-9a-zA-Z_]+", "_", name)
    name = name.strip("_")
    return name[:120] if name else "unknown"


def unique_keep_order(items):
    seen = set()
    out = []
    for x in items:
        if x is None:
            continue
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def find_first_existing_col(columns, candidates):
    col_list = list(columns)
    lower_map = {str(c).lower().strip(): c for c in col_list}

    for cand in candidates:
        key = str(cand).lower().strip()
        if key in lower_map:
            return lower_map[key]

    for cand in candidates:
        key = str(cand).lower().strip()
        for c in col_list:
            if key in str(c).lower().strip():
                return c

    return None


def get_label_related_cols(columns):
    label_cols = []
    targets = {"label", "sublabel", "sublabelcat"}

    for c in columns:
        c_lower = str(c).lower().strip()
        if c_lower in targets:
            label_cols.append(c)

    return label_cols


def normalize_str_value(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def build_benign_mask(df):
    """
    Phase-1 理论上为纯良性。
    这里仍根据 label/subLabel/subLabelCat 做安全过滤：
        - label 必须表现为 benign/normal/0/空值；
        - subLabel/subLabelCat 中不能出现 attack/malicious/apt 等攻击词。
    """
    if not ENABLE_BENIGN_FILTER:
        return pd.Series(True, index=df.index)

    mask = pd.Series(True, index=df.index)

    label_cols = get_label_related_cols(df.columns)

    benign_values = {
        "",
        "0",
        "0.0",
        "benign",
        "benigntraffic",
        "benign traffic",
        "normal",
        "clean",
        "background",
        "none",
        "nan"
    }

    attack_keywords = [
        "attack",
        "malicious",
        "apt",
        "recon",
        "scan",
        "dos",
        "ddos",
        "injection",
        "bruteforce",
        "brute",
        "exploit",
        "backdoor",
        "command",
        "control",
        "c2",
        "exfiltration",
        "privilege",
        "lateral"
    ]

    main_label_col = None
    for c in label_cols:
        if str(c).lower().strip() == "label":
            main_label_col = c
            break

    if main_label_col is not None:
        label_norm = df[main_label_col].apply(normalize_str_value)
        label_benign_mask = (
            label_norm.isin(benign_values)
            | label_norm.str.contains("benign", regex=False, na=False)
        )
        mask &= label_benign_mask

    for col in label_cols:
        s = df[col].apply(normalize_str_value)
        for kw in attack_keywords:
            mask &= ~s.str.contains(kw, regex=False, na=False)

    return mask


def to_epoch_seconds(series: pd.Series, col_name: str = "") -> pd.Series:
    """
    将各种时间格式统一转换成秒。
    支持：
        - epoch seconds
        - epoch milliseconds
        - epoch microseconds
        - epoch nanoseconds
        - datetime string
        - pandas datetime
    """
    s = series.copy()

    if pd.api.types.is_datetime64_any_dtype(s):
        dt = pd.to_datetime(s, errors="coerce", utc=True)
        out = pd.Series(np.nan, index=s.index, dtype="float64")
        mask = dt.notna()
        if mask.any():
            out.loc[mask] = dt.loc[mask].astype("int64") / 1e9
        return out

    numeric = pd.to_numeric(s, errors="coerce")
    numeric_ratio = numeric.notna().mean()

    if numeric_ratio >= 0.80:
        valid_abs = numeric.dropna().abs()
        if len(valid_abs) == 0:
            return pd.Series(np.nan, index=s.index, dtype="float64")

        med = valid_abs.median()

        if med > 1e17:
            factor = 1e9
        elif med > 1e14:
            factor = 1e6
        elif med > 1e11:
            factor = 1e3
        else:
            factor = 1.0

        return numeric.astype("float64") / factor

    dt = pd.to_datetime(s, errors="coerce", utc=True)
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    mask = dt.notna()
    if mask.any():
        out.loc[mask] = dt.loc[mask].astype("int64") / 1e9

    return out


def make_window_id(seconds_series: pd.Series) -> pd.Series:
    return np.floor(seconds_series.astype("float64") / TIME_WINDOW_SEC).astype("int64")


def concat_groupby_sum(parts):
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, axis=0).groupby(level=0).sum()


def make_datetime_strings(seconds_array):
    seconds_array = np.asarray(seconds_array, dtype="float64")
    if len(seconds_array) == 0:
        return pd.Series([], dtype=str)

    med = np.nanmedian(seconds_array)
    if np.isfinite(med) and 1e8 < med < 5e9:
        return pd.Series(pd.to_datetime(seconds_array, unit="s", utc=True)).astype(str)

    return pd.Series([""] * len(seconds_array), dtype=str)


def format_time_range(min_sec, max_sec):
    if min_sec is None or max_sec is None:
        return "N/A"

    if not np.isfinite(min_sec) or not np.isfinite(max_sec):
        return "N/A"

    if 1e8 < min_sec < 5e9 and 1e8 < max_sec < 5e9:
        try:
            start = pd.to_datetime(min_sec, unit="s", utc=True)
            end = pd.to_datetime(max_sec, unit="s", utc=True)
            return f"{min_sec:.3f} -> {max_sec:.3f} | {start} -> {end}"
        except Exception:
            return f"{min_sec:.3f} -> {max_sec:.3f}"

    return f"{min_sec:.3f} -> {max_sec:.3f} | relative seconds"


def read_header_columns(path):
    return list(pd.read_csv(path, nrows=0).columns)


# =====================================================
# 4. 读取恶意阶段的 85 维特征列
# =====================================================
def load_reference_feature_columns():
    if not os.path.exists(FILE_REF_FEATURE_COLUMNS):
        raise FileNotFoundError(
            f"找不到恶意抽取阶段生成的特征列文件: {FILE_REF_FEATURE_COLUMNS}\n"
            f"请先运行恶意抽取代码，生成 cicapt_feature_columns.csv。"
        )

    df = pd.read_csv(FILE_REF_FEATURE_COLUMNS)

    if "feature_name" in df.columns:
        feature_cols = df["feature_name"].dropna().astype(str).tolist()
    else:
        feature_cols = df.iloc[:, 0].dropna().astype(str).tolist()

    feature_cols = [c.strip() for c in feature_cols if str(c).strip()]

    if len(feature_cols) != len(set(feature_cols)):
        duplicated = pd.Series(feature_cols).value_counts()
        duplicated = duplicated[duplicated > 1]
        raise ValueError(f"特征列存在重复项: {duplicated.to_dict()}")

    if len(feature_cols) != EXPECTED_FEATURE_DIM:
        raise ValueError(
            f"参考特征维度不是 {EXPECTED_FEATURE_DIM}，当前为 {len(feature_cols)}。\n"
            f"请确认 cicapt_feature_columns.csv 是否来自前面恶意抽取代码。"
        )

    logger.info(f"成功读取参考特征列: {FILE_REF_FEATURE_COLUMNS}")
    logger.info(f"参考特征维度: {len(feature_cols)}")

    return feature_cols


def split_reference_features(feature_cols):
    network_features = []
    provenance_features = []

    for c in feature_cols:
        if str(c).startswith("prov_"):
            provenance_features.append(c)
        else:
            network_features.append(c)

    logger.info(f"参考网络特征数量: {len(network_features)}")
    logger.info(f"参考 provenance 特征数量: {len(provenance_features)}")

    return network_features, provenance_features


# =====================================================
# 5. 分块处理 Phase-1 NetworkData
# =====================================================
def process_network_data(ref_network_features):
    if not os.path.exists(FILE_NETWORK):
        logger.warning(f"未找到网络数据文件: {FILE_NETWORK}")
        return pd.DataFrame(), None, None

    if len(ref_network_features) == 0:
        logger.warning("参考特征中没有网络特征，跳过 NetworkData。")
        return pd.DataFrame(), None, None

    logger.info("=" * 80)
    logger.info(f"开始分块处理 Phase-1 网络数据: {FILE_NETWORK}")

    header_cols = read_header_columns(FILE_NETWORK)
    ts_col = find_first_existing_col(header_cols, NETWORK_TS_CANDIDATES)

    if ts_col is None:
        raise ValueError(f"网络数据中找不到时间戳列。当前列名为: {header_cols}")

    need_net_record_count = "net_record_count" in ref_network_features
    network_mean_features = [c for c in ref_network_features if c != "net_record_count"]

    label_cols = get_label_related_cols(header_cols)

    read_cols = unique_keep_order(
        [ts_col]
        + [c for c in network_mean_features if c in header_cols]
        + label_cols
    )

    missing_raw_cols = [c for c in network_mean_features if c not in header_cols]
    if missing_raw_cols:
        logger.warning(f"Phase-1 网络数据缺少以下参考特征，将填充为 0: {missing_raw_cols}")

    logger.info(f"网络时间戳列: {ts_col}")
    logger.info(f"网络分块读取列数量: {len(read_cols)}")
    logger.info(f"网络均值特征数量: {len(network_mean_features)}")

    sum_parts = []
    count_parts = []
    record_count_parts = []

    net_min_sec = np.inf
    net_max_sec = -np.inf

    for chunk_id, chunk in enumerate(
        pd.read_csv(FILE_NETWORK, usecols=read_cols, chunksize=CHUNK_SIZE, low_memory=False),
        start=1
    ):
        logger.info(f"  -> 网络数据第 {chunk_id} 块，行数: {len(chunk)}")

        seconds = to_epoch_seconds(chunk[ts_col], ts_col)
        valid_time = seconds.notna()

        if not valid_time.any():
            logger.warning(f"网络数据第 {chunk_id} 块没有有效时间戳，跳过。")
            continue

        chunk = chunk.loc[valid_time].copy()
        seconds = seconds.loc[valid_time]

        benign_mask = build_benign_mask(chunk)
        if not benign_mask.any():
            logger.warning(f"网络数据第 {chunk_id} 块没有通过良性过滤的记录，跳过。")
            continue

        chunk = chunk.loc[benign_mask].copy()
        seconds = seconds.loc[benign_mask]

        net_min_sec = min(net_min_sec, float(seconds.min()))
        net_max_sec = max(net_max_sec, float(seconds.max()))

        chunk["window_id"] = make_window_id(seconds)

        if network_mean_features:
            data = pd.DataFrame(index=chunk.index)

            for feat in network_mean_features:
                if feat in chunk.columns:
                    values = pd.to_numeric(chunk[feat], errors="coerce")
                    values = values.replace([np.inf, -np.inf], np.nan)
                    data[feat] = values
                else:
                    data[feat] = 0.0

            tmp = pd.concat([chunk["window_id"], data], axis=1)
            g = tmp.groupby("window_id", sort=False)

            sum_df = g[network_mean_features].sum(min_count=1).fillna(0.0)
            count_df = g[network_mean_features].count()

            sum_parts.append(sum_df)
            count_parts.append(count_df)

        if need_net_record_count:
            record_count_df = (
                chunk.groupby("window_id", sort=False)
                .size()
                .rename("net_record_count")
                .to_frame()
            )
            record_count_parts.append(record_count_df)

    result_parts = []

    if network_mean_features and sum_parts:
        logger.info("正在执行网络数据全局 sum/count 聚合...")

        total_sum = concat_groupby_sum(sum_parts)
        total_count = concat_groupby_sum(count_parts)

        total_sum = total_sum.reindex(columns=network_mean_features, fill_value=0.0)
        total_count = total_count.reindex(columns=network_mean_features, fill_value=0.0)

        net_mean = total_sum.div(total_count.replace(0, np.nan)).fillna(0.0)
        result_parts.append(net_mean)

    if need_net_record_count and record_count_parts:
        total_record_count = concat_groupby_sum(record_count_parts)
        result_parts.append(total_record_count)

    if not result_parts:
        logger.warning("网络数据没有产生任何有效窗口。")
        return pd.DataFrame(), None, None

    net_agg = pd.concat(result_parts, axis=1).fillna(0.0)

    for c in ref_network_features:
        if c not in net_agg.columns:
            net_agg[c] = 0.0

    net_agg = net_agg[ref_network_features]
    net_agg.index.name = "window_id"
    net_agg = net_agg.reset_index()
    net_agg["window_id"] = net_agg["window_id"].astype("int64")

    logger.info(f"网络数据处理完成，窗口数量: {len(net_agg)}")
    logger.info(f"网络时间范围: {format_time_range(net_min_sec, net_max_sec)}")
    logger.info("=" * 80)

    return net_agg, net_min_sec, net_max_sec


# =====================================================
# 6. 分块处理 Phase-1 Provenance
# =====================================================
def detect_prov_text_columns_from_header(columns, time_cols):
    selected = []
    time_cols_lower = {str(c).lower().strip() for c in time_cols}

    for c in columns:
        c_lower = str(c).lower().strip()
        if c_lower in time_cols_lower:
            continue

        if any(h in c_lower for h in PROV_TEXT_HINTS):
            selected.append(c)

    return selected[:8]


def process_provenance_data(ref_prov_features):
    if not os.path.exists(FILE_PROVENANCE):
        logger.warning(f"未找到溯源数据文件: {FILE_PROVENANCE}")
        return pd.DataFrame(), None, None

    if len(ref_prov_features) == 0:
        logger.warning("参考特征中没有 provenance 特征，跳过 Provenance。")
        return pd.DataFrame(), None, None

    logger.info("=" * 80)
    logger.info(f"开始分块处理 Phase-1 溯源数据: {FILE_PROVENANCE}")

    header_cols = read_header_columns(FILE_PROVENANCE)

    time_cols = []
    for cand in PROVENANCE_TIME_CANDIDATES:
        col = find_first_existing_col(header_cols, [cand])
        if col is not None and col not in time_cols:
            time_cols.append(col)

    if len(time_cols) == 0:
        raise ValueError(f"溯源数据中找不到时间列。当前列名为: {header_cols}")

    need_event_count = "prov_event_count" in ref_prov_features
    keyword_features = [c for c in ref_prov_features if c in KEYWORD_FEATURES]
    num_features = [c for c in ref_prov_features if c.startswith(PROV_NUM_PREFIX)]
    unique_features = [c for c in ref_prov_features if c.startswith(PROV_UNIQUE_PREFIX)]

    safe_to_col = {safe_name(c): c for c in header_cols}

    num_feature_to_col = {}
    for feat in num_features:
        suffix = feat[len(PROV_NUM_PREFIX):]
        if suffix in safe_to_col:
            num_feature_to_col[feat] = safe_to_col[suffix]

    unique_feature_to_col = {}
    for feat in unique_features:
        suffix = feat[len(PROV_UNIQUE_PREFIX):]
        if suffix in safe_to_col:
            unique_feature_to_col[feat] = safe_to_col[suffix]

    text_cols = detect_prov_text_columns_from_header(header_cols, time_cols)
    label_cols = get_label_related_cols(header_cols)

    read_cols = unique_keep_order(
        time_cols
        + text_cols
        + list(num_feature_to_col.values())
        + list(unique_feature_to_col.values())
        + label_cols
    )

    logger.info(f"溯源时间候选列: {time_cols}")
    logger.info(f"溯源关键词统计文本列: {text_cols}")
    logger.info(f"溯源数值均值特征数量: {len(num_features)}")
    logger.info(f"溯源唯一实体特征数量: {len(unique_features)}")
    logger.info(f"溯源分块读取列数量: {len(read_cols)}")

    missing_num = [f for f in num_features if f not in num_feature_to_col]
    missing_unique = [f for f in unique_features if f not in unique_feature_to_col]

    if missing_num:
        logger.warning(f"Phase-1 Provenance 缺少以下数值参考特征，将填充为 0: {missing_num}")

    if missing_unique:
        logger.warning(f"Phase-1 Provenance 缺少以下唯一值参考特征，将填充为 0: {missing_unique}")

    count_parts = []
    num_sum_parts = []
    num_count_parts = []

    unique_sets = defaultdict(lambda: defaultdict(set))

    prov_min_sec = np.inf
    prov_max_sec = -np.inf

    for chunk_id, chunk in enumerate(
        pd.read_csv(FILE_PROVENANCE, usecols=read_cols, chunksize=CHUNK_SIZE, low_memory=False),
        start=1
    ):
        logger.info(f"  -> 溯源数据第 {chunk_id} 块，行数: {len(chunk)}")

        event_seconds = pd.Series(np.nan, index=chunk.index, dtype="float64")

        for tc in time_cols:
            if tc in chunk.columns:
                sec = to_epoch_seconds(chunk[tc], tc)
                event_seconds = event_seconds.combine_first(sec)

        valid_time = event_seconds.notna()

        if not valid_time.any():
            logger.warning(f"溯源数据第 {chunk_id} 块没有有效时间戳，跳过。")
            continue

        chunk = chunk.loc[valid_time].copy()
        event_seconds = event_seconds.loc[valid_time]

        benign_mask = build_benign_mask(chunk)
        if not benign_mask.any():
            logger.warning(f"溯源数据第 {chunk_id} 块没有通过良性过滤的记录，跳过。")
            continue

        chunk = chunk.loc[benign_mask].copy()
        event_seconds = event_seconds.loc[benign_mask]

        prov_min_sec = min(prov_min_sec, float(event_seconds.min()))
        prov_max_sec = max(prov_max_sec, float(event_seconds.max()))

        chunk["window_id"] = make_window_id(event_seconds)

        g = chunk.groupby("window_id", sort=False)
        group_size = g.size()

        # 6.1 基础事件计数与关键词事件计数
        count_df = pd.DataFrame(index=group_size.index)

        if need_event_count:
            count_df["prov_event_count"] = group_size.astype("float64")

        if keyword_features:
            for feat_name in keyword_features:
                pattern = KEYWORD_FEATURES[feat_name]
                mask = pd.Series(False, index=chunk.index)

                for col in text_cols:
                    if col in chunk.columns:
                        col_text = chunk[col].astype("string").str.lower()
                        mask = mask | col_text.str.contains(pattern, regex=True, na=False)

                tmp = pd.DataFrame({
                    "window_id": chunk["window_id"].values,
                    feat_name: mask.astype("int8").values
                })

                kw_count = tmp.groupby("window_id", sort=False)[feat_name].sum()
                count_df = count_df.join(kw_count, how="outer")

        if len(count_df.columns) > 0:
            count_df = count_df.fillna(0.0)
            count_parts.append(count_df)

        # 6.2 provenance 数值列均值
        if num_features:
            num_data = pd.DataFrame(index=chunk.index)

            for feat in num_features:
                raw_col = num_feature_to_col.get(feat, None)

                if raw_col is not None and raw_col in chunk.columns:
                    values = pd.to_numeric(chunk[raw_col], errors="coerce")
                    values = values.replace([np.inf, -np.inf], np.nan)
                    num_data[feat] = values
                else:
                    num_data[feat] = 0.0

            num_tmp = pd.concat([chunk["window_id"], num_data], axis=1)
            ng = num_tmp.groupby("window_id", sort=False)

            nsum = ng[num_features].sum(min_count=1).fillna(0.0)
            ncnt = ng[num_features].count()

            num_sum_parts.append(nsum)
            num_count_parts.append(ncnt)

        # 6.3 provenance 唯一值统计
        if unique_features:
            for feat in unique_features:
                raw_col = unique_feature_to_col.get(feat, None)

                if raw_col is None or raw_col not in chunk.columns:
                    continue

                grouped = chunk.groupby("window_id", sort=False)[raw_col].agg(
                    lambda x: set(x.dropna().astype(str))
                )

                for wid, val_set in grouped.items():
                    unique_sets[feat][int(wid)].update(val_set)

    result_parts = []

    if count_parts:
        prov_count_all = concat_groupby_sum(count_parts)
        result_parts.append(prov_count_all)

    if num_sum_parts and num_count_parts:
        prov_num_sum = concat_groupby_sum(num_sum_parts)
        prov_num_count = concat_groupby_sum(num_count_parts)

        prov_num_sum = prov_num_sum.reindex(columns=num_features, fill_value=0.0)
        prov_num_count = prov_num_count.reindex(columns=num_features, fill_value=0.0)

        prov_num_mean = prov_num_sum.div(prov_num_count.replace(0, np.nan)).fillna(0.0)
        result_parts.append(prov_num_mean)

    if unique_sets:
        all_wids = set()
        for feat, wid_dict in unique_sets.items():
            all_wids.update(wid_dict.keys())

        unique_df = pd.DataFrame(index=sorted(all_wids))

        for feat in unique_features:
            unique_df[feat] = 0.0
            wid_dict = unique_sets.get(feat, {})
            for wid, val_set in wid_dict.items():
                unique_df.loc[wid, feat] = float(len(val_set))

        result_parts.append(unique_df)

    if not result_parts:
        logger.warning("溯源数据没有产生任何有效窗口。")
        return pd.DataFrame(), None, None

    prov_agg = pd.concat(result_parts, axis=1).fillna(0.0)

    for c in ref_prov_features:
        if c not in prov_agg.columns:
            prov_agg[c] = 0.0

    prov_agg = prov_agg[ref_prov_features]
    prov_agg.index.name = "window_id"
    prov_agg = prov_agg.reset_index()
    prov_agg["window_id"] = prov_agg["window_id"].astype("int64")

    logger.info(f"溯源数据处理完成，窗口数量: {len(prov_agg)}")
    logger.info(f"溯源时间范围: {format_time_range(prov_min_sec, prov_max_sec)}")
    logger.info("=" * 80)

    return prov_agg, prov_min_sec, prov_max_sec


# =====================================================
# 7. 选择固定 1890 个良性窗口
# =====================================================
def select_target_rows(merged):
    total = len(merged)

    if total < TARGET_ROWS:
        raise ValueError(
            f"良性窗口数量不足，无法抽取 {TARGET_ROWS} 行。当前只有 {total} 行。"
        )

    if SAMPLE_MODE == "head":
        positions = np.arange(TARGET_ROWS, dtype=np.int64)

    elif SAMPLE_MODE == "tail":
        positions = np.arange(total - TARGET_ROWS, total, dtype=np.int64)

    elif SAMPLE_MODE == "uniform":
        positions = np.linspace(0, total - 1, TARGET_ROWS, dtype=np.int64)

    else:
        raise ValueError(f"不支持的 SAMPLE_MODE: {SAMPLE_MODE}")

    selected = merged.iloc[positions].copy()
    selected = selected.sort_values("window_id").reset_index(drop=True)

    return selected


# =====================================================
# 8. 验证与保存
# =====================================================
def validate_against_attack(obs_matrix):
    if obs_matrix.shape != (TARGET_ROWS, EXPECTED_FEATURE_DIM):
        raise ValueError(
            f"良性特征矩阵 shape 错误，期望 {(TARGET_ROWS, EXPECTED_FEATURE_DIM)}，"
            f"实际 {obs_matrix.shape}"
        )

    if os.path.exists(FILE_ATTACK_FEATURES):
        attack_matrix = np.load(FILE_ATTACK_FEATURES, mmap_mode="r")

        logger.info("恶意特征文件维度校验:")
        logger.info(f"  attack feature path: {FILE_ATTACK_FEATURES}")
        logger.info(f"  attack feature shape: {attack_matrix.shape}")

        if attack_matrix.ndim != 2:
            raise ValueError(f"恶意特征矩阵应为 2D，实际 shape={attack_matrix.shape}")

        if attack_matrix.shape[1] != obs_matrix.shape[1]:
            raise ValueError(
                f"良性/恶意特征维度不一致: "
                f"benign_dim={obs_matrix.shape[1]}, attack_dim={attack_matrix.shape[1]}"
            )

        logger.info("良性/恶意特征维度校验通过。")
    else:
        logger.warning(f"未找到恶意特征文件 {FILE_ATTACK_FEATURES}，跳过 npy 维度交叉校验。")


def save_outputs(selected, raw_matrix, obs_matrix, label_matrix, feature_cols, net_range, prov_range):
    np.save(OUTPUT_FEATURES_RAW, raw_matrix)
    np.save(OUTPUT_FEATURES, obs_matrix)

    if SAVE_ZERO_LABELS:
        np.save(OUTPUT_LABELS, label_matrix)

    pd.Series(feature_cols, name="feature_name").to_csv(OUTPUT_FEATURE_COLUMNS, index=False)

    label_cols = [
        "risk_proxy_0_10",
        "service_pressure_proxy_0_5",
        "is_attack",
        "attack_stage_id"
    ]
    pd.Series(label_cols, name="label_name").to_csv(OUTPUT_LABEL_COLUMNS, index=False)

    window_index_cols = [
        "window_id",
        "window_ts",
        "window_datetime_utc"
    ]
    selected[window_index_cols].to_csv(OUTPUT_WINDOW_INDEX, index=False)

    nan_count = int(np.isnan(obs_matrix).sum())
    inf_count = int(np.isinf(obs_matrix).sum())
    nonzero_ratio = float(np.count_nonzero(obs_matrix) / max(obs_matrix.size, 1))

    summary = {
        "time_window_sec": TIME_WINDOW_SEC,
        "chunk_size": CHUNK_SIZE,
        "target_rows": TARGET_ROWS,
        "sample_mode": SAMPLE_MODE,
        "apply_zscore": APPLY_ZSCORE,
        "feature_shape": list(obs_matrix.shape),
        "label_shape": list(label_matrix.shape),
        "feature_dim": int(obs_matrix.shape[1]),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "nonzero_ratio": nonzero_ratio,
        "is_pure_benign": True,
        "zero_label_saved": bool(SAVE_ZERO_LABELS),
        "reference_feature_file": FILE_REF_FEATURE_COLUMNS,
        "network_time_range": {
            "min": None if net_range[0] is None else float(net_range[0]),
            "max": None if net_range[1] is None else float(net_range[1]),
            "readable": format_time_range(net_range[0], net_range[1])
        },
        "provenance_time_range": {
            "min": None if prov_range[0] is None else float(prov_range[0]),
            "max": None if prov_range[1] is None else float(prov_range[1]),
            "readable": format_time_range(prov_range[0], prov_range[1])
        }
    }

    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("良性抽取文件已保存:")
    logger.info(f"  {OUTPUT_FEATURES}, shape={obs_matrix.shape}")
    logger.info(f"  {OUTPUT_FEATURES_RAW}, shape={raw_matrix.shape}")

    if SAVE_ZERO_LABELS:
        logger.info(f"  {OUTPUT_LABELS}, shape={label_matrix.shape}")

    logger.info(f"  {OUTPUT_FEATURE_COLUMNS}")
    logger.info(f"  {OUTPUT_LABEL_COLUMNS}")
    logger.info(f"  {OUTPUT_WINDOW_INDEX}")
    logger.info(f"  {OUTPUT_SUMMARY}")


# =====================================================
# 9. 主流程
# =====================================================
def process_data():
    logger.info("=" * 80)
    logger.info("开始处理 CICAPT-IIoT Phase-1 纯良性数据")
    logger.info(f"时间窗口: {TIME_WINDOW_SEC} 秒")
    logger.info(f"分块大小: {CHUNK_SIZE} 行")
    logger.info(f"目标输出行数: {TARGET_ROWS}")
    logger.info(f"期望特征维度: {EXPECTED_FEATURE_DIM}")
    logger.info(f"抽样模式: {SAMPLE_MODE}")
    logger.info(f"是否过滤良性标签: {ENABLE_BENIGN_FILTER}")
    logger.info("=" * 80)

    # Step 1: 读取恶意阶段生成的 85 维特征列
    feature_cols = load_reference_feature_columns()
    ref_network_features, ref_prov_features = split_reference_features(feature_cols)

    # Step 2: 分块处理 Phase-1 NetworkData
    net_agg, net_min, net_max = process_network_data(ref_network_features)

    # Step 3: 分块处理 Phase-1 Provenance
    prov_agg, prov_min, prov_max = process_provenance_data(ref_prov_features)

    if net_agg.empty and prov_agg.empty:
        raise RuntimeError("Phase-1 网络数据和溯源数据均为空，无法生成良性 npy。")

    # Step 4: 多源时间窗口融合
    logger.info("=" * 80)
    logger.info("开始融合 Phase-1 网络数据与溯源数据...")

    dfs = []
    if not net_agg.empty:
        dfs.append(net_agg)

    if not prov_agg.empty:
        dfs.append(prov_agg)

    if len(dfs) == 1:
        merged = dfs[0].copy()
    else:
        merged = reduce(
            lambda left, right: pd.merge(left, right, on="window_id", how="outer"),
            dfs
        )

    merged["window_id"] = merged["window_id"].astype("int64")
    merged = merged.sort_values("window_id").reset_index(drop=True)

    if CREATE_CONTINUOUS_TIMELINE:
        min_wid = int(merged["window_id"].min())
        max_wid = int(merged["window_id"].max())
        num_windows = max_wid - min_wid + 1

        if num_windows <= MAX_CONTINUOUS_WINDOWS:
            logger.info(f"生成连续时间轴，窗口数: {num_windows}")
            grid = pd.DataFrame({
                "window_id": np.arange(min_wid, max_wid + 1, dtype=np.int64)
            })
            merged = pd.merge(grid, merged, on="window_id", how="left")
        else:
            logger.warning(
                f"连续窗口数 {num_windows} 超过 MAX_CONTINUOUS_WINDOWS={MAX_CONTINUOUS_WINDOWS}，"
                f"将保留稀疏时间轴。"
            )

    merged["window_ts"] = merged["window_id"].astype("float64") * TIME_WINDOW_SEC
    merged["window_datetime_utc"] = make_datetime_strings(merged["window_ts"].values)

    # Step 5: 严格按恶意阶段 feature_cols 对齐
    for c in feature_cols:
        if c not in merged.columns:
            merged[c] = 0.0

    merged[feature_cols] = merged[feature_cols].fillna(0.0)

    logger.info(f"融合后良性窗口数量: {len(merged)}")
    logger.info(f"融合后特征数量: {len(feature_cols)}")
    logger.info("=" * 80)

    # Step 6: 控制为 1890 行
    selected = select_target_rows(merged)

    logger.info("良性窗口选择完成:")
    logger.info(f"  原始良性窗口数: {len(merged)}")
    logger.info(f"  选择后窗口数: {len(selected)}")
    logger.info(f"  选择模式: {SAMPLE_MODE}")

    # Step 7: 构建特征矩阵
    feature_df = selected[feature_cols].apply(pd.to_numeric, errors="coerce")
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    raw_matrix = feature_df.values.astype("float32")
    raw_matrix = np.nan_to_num(raw_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    if APPLY_ZSCORE:
        mean = raw_matrix.mean(axis=0, keepdims=True)
        std = raw_matrix.std(axis=0, keepdims=True)
        std[std < EPS] = 1.0
        obs_matrix = ((raw_matrix - mean) / std).astype("float32")
    else:
        obs_matrix = raw_matrix.astype("float32")

    # Step 8: 构建全 0 良性标签矩阵，兼容恶意抽取的 4 维 label
    label_matrix = np.zeros((len(selected), 4), dtype="float32")

    # Step 9: 校验 shape 和恶意特征维度
    validate_against_attack(obs_matrix)

    # Step 10: 保存
    save_outputs(
        selected=selected,
        raw_matrix=raw_matrix,
        obs_matrix=obs_matrix,
        label_matrix=label_matrix,
        feature_cols=feature_cols,
        net_range=(None if net_min is None or not np.isfinite(net_min) else net_min,
                   None if net_max is None or not np.isfinite(net_max) else net_max),
        prov_range=(None if prov_min is None or not np.isfinite(prov_min) else prov_min,
                    None if prov_max is None or not np.isfinite(prov_max) else prov_max)
    )

    logger.info("=" * 80)
    logger.info("✅ Phase-1 纯良性数据抽取完成")
    logger.info(f"输出良性特征文件: {OUTPUT_FEATURES}")
    logger.info(f"输出良性标签文件: {OUTPUT_LABELS}")
    logger.info(f"良性特征矩阵 Shape: {obs_matrix.shape}")
    logger.info(f"良性标签矩阵 Shape: {label_matrix.shape}")
    logger.info("=" * 80)


if __name__ == "__main__":
    process_data()