# -*- coding: utf-8 -*-
"""
CICAPT-IIoT Data Ingestion Pipeline V2.0
Target:
    1. Keep all network numeric dimensions.
    2. Fuse network data, provenance data, and attack_info.
    3. Generate window-level .npy sequence for RL decision training.
    4. Provide verification files for attack-window alignment.

Input:
    phase2_NetworkData.csv
    Phase2_Provenance.csv
    attack_info.csv

Output:
    cicapt_ot_sequence_malicious.npy
    cicapt_ot_labels_malicious.npy
    cicapt_feature_columns.csv
    cicapt_label_columns.csv
    cicapt_window_index.csv
    verify_attack_windows.csv
    verify_attack_neighborhood.csv
    top_feature_differences.csv
    extraction_summary.json
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
CHUNK_SIZE = 1_000_000

FILE_NETWORK = "phase2_NetworkData.csv"
FILE_PROVENANCE = "Phase2_Provenance.csv"
FILE_ATTACK = "attack_info.csv"

OUTPUT_FEATURES = "cicapt_ot_sequence_malicious.npy"
OUTPUT_FEATURES_RAW = "cicapt_ot_sequence_malicious_raw.npy"
OUTPUT_LABELS = "cicapt_ot_labels_malicious.npy"

# 是否生成完整连续时间轴。建议 True，便于 RL 按时间步训练。
CREATE_CONTINUOUS_TIMELINE = True
MAX_CONTINUOUS_WINDOWS = 2_000_000

# 如果 attack_info 只有一个攻击时间点，没有结束时间，则默认认为攻击覆盖一个窗口。
POINT_ATTACK_DURATION_SEC = TIME_WINDOW_SEC
MIN_ATTACK_INTERVAL_SEC = TIME_WINDOW_SEC

# 是否对最终特征做 z-score。若你的主方法内部已有归一化，这里建议保持 False。
APPLY_ZSCORE = False
EPS = 1e-8

# provenance 高基数唯一值统计可能占内存，默认开启，但只选少量关键列。
ENABLE_EXACT_PROV_NUNIQUE = True
PROV_NUNIQUE_MAX_COLS = 4
MAX_PROV_TEXT_COLS = 8

NETWORK_TS_CANDIDATES = [
    "ts", "timestamp", "time", "Time", "datetime", "date_time"
]

PROVENANCE_TIME_CANDIDATES = [
    "time", "seen time", "seen_time", "timestamp", "ts", "event_time", "datetime"
]

ATTACK_START_CANDIDATES = [
    "Time of Attack", "time of attack", "attack_time", "Attack Time",
    "Start Time", "start_time", "start", "begin", "Begin Time",
    "timestamp", "time", "ts"
]

ATTACK_END_CANDIDATES = [
    "End Time", "end_time", "end", "finish", "Finish Time",
    "Attack End", "attack_end", "stop_time"
]

ATTACK_DURATION_CANDIDATES = [
    "Duration", "duration", "duration_sec", "Duration_sec", "attack_duration"
]

ATTACK_STAGE_CANDIDATES = [
    "phase", "Phase", "stage", "Stage", "attack_phase", "Attack Phase",
    "subLabelCat", "subLabel", "label"
]

ATTACK_TYPE_CANDIDATES = [
    "attack", "Attack", "attack_type", "Attack Type", "type", "Type",
    "technique", "Technique", "subLabel", "subLabelCat", "label"
]

EXCLUDE_NETWORK_COLS = {
    "ts", "timestamp", "time",
    "source ip", "destination ip", "src ip", "dst ip",
    "protocol_name", "protocol name",
    "label", "sublabel", "sublabelcat"
}

PROV_TEXT_HINTS = [
    "type", "event", "operation", "action", "relation", "predicate",
    "object", "subject", "process", "file", "path", "socket",
    "cmd", "command", "name"
]

PROV_UNIQUE_HINTS = [
    "subject", "object", "process", "file", "path", "socket",
    "src", "dst", "source", "destination", "actor", "target",
    "pid", "uid", "node", "entity"
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

# =====================================================
# 2. 日志初始化
# =====================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("CICAPT_Extractor")


# =====================================================
# 3. 通用工具函数
# =====================================================
def safe_name(name: str) -> str:
    name = str(name)
    name = re.sub(r"[^0-9a-zA-Z_]+", "_", name)
    name = name.strip("_")
    return name[:120] if name else "unknown"


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
            factor = 1e9       # ns
        elif med > 1e14:
            factor = 1e6       # us
        elif med > 1e11:
            factor = 1e3       # ms
        else:
            factor = 1.0       # seconds or relative seconds

        return numeric.astype("float64") / factor

    dt = pd.to_datetime(s, errors="coerce", utc=True)
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    mask = dt.notna()
    if mask.any():
        out.loc[mask] = dt.loc[mask].astype("int64") / 1e9

    return out


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


def detect_numeric_columns(chunk: pd.DataFrame, exclude_lower_set):
    numeric_cols = []

    for c in chunk.columns:
        c_lower = str(c).lower().strip()
        if c_lower in exclude_lower_set:
            continue

        sample = chunk[c].dropna().head(3000)
        if len(sample) == 0:
            continue

        converted = pd.to_numeric(sample, errors="coerce")
        valid_ratio = converted.notna().mean()

        if valid_ratio >= 0.95:
            numeric_cols.append(c)

    return numeric_cols


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


# =====================================================
# 4. 读取 attack_info.csv，构造攻击区间
# =====================================================
def load_attack_intervals():
    if not os.path.exists(FILE_ATTACK):
        raise FileNotFoundError(f"找不到攻击真值文件: {FILE_ATTACK}")

    logger.info(f"正在读取攻击真值文件: {FILE_ATTACK}")
    attack_df = pd.read_csv(FILE_ATTACK, low_memory=False)

    start_col = find_first_existing_col(attack_df.columns, ATTACK_START_CANDIDATES)
    end_col = find_first_existing_col(attack_df.columns, ATTACK_END_CANDIDATES)
    duration_col = find_first_existing_col(attack_df.columns, ATTACK_DURATION_CANDIDATES)
    stage_col = find_first_existing_col(attack_df.columns, ATTACK_STAGE_CANDIDATES)
    type_col = find_first_existing_col(attack_df.columns, ATTACK_TYPE_CANDIDATES)

    if start_col is None:
        raise ValueError(
            f"attack_info.csv 中找不到攻击开始时间列。当前列名为: {list(attack_df.columns)}"
        )

    logger.info(f"攻击开始时间列: {start_col}")
    logger.info(f"攻击结束时间列: {end_col if end_col else '未找到，使用点攻击窗口'}")
    logger.info(f"攻击阶段列: {stage_col if stage_col else '未找到'}")
    logger.info(f"攻击类型列: {type_col if type_col else '未找到'}")

    start_sec = to_epoch_seconds(attack_df[start_col], start_col)

    if end_col is not None:
        end_sec = to_epoch_seconds(attack_df[end_col], end_col)
    else:
        end_sec = pd.Series(np.nan, index=attack_df.index, dtype="float64")

    if end_col is None and duration_col is not None:
        duration = pd.to_numeric(attack_df[duration_col], errors="coerce")
        end_sec = start_sec + duration.fillna(POINT_ATTACK_DURATION_SEC)
        logger.info(f"使用持续时间列构造攻击结束时间: {duration_col}")

    if end_col is None and duration_col is None:
        end_sec = start_sec + POINT_ATTACK_DURATION_SEC

    tmp = pd.DataFrame({
        "start_sec": start_sec,
        "end_sec": end_sec
    })

    valid = tmp["start_sec"].notna()
    tmp = tmp.loc[valid].copy()
    original_valid_index = tmp.index

    tmp["end_sec"] = tmp["end_sec"].fillna(tmp["start_sec"] + POINT_ATTACK_DURATION_SEC)

    bad_interval = tmp["end_sec"] <= tmp["start_sec"]
    if bad_interval.any():
        tmp.loc[bad_interval, "end_sec"] = tmp.loc[bad_interval, "start_sec"] + MIN_ATTACK_INTERVAL_SEC

    def encode_text_column(col, default_name):
        if col is None:
            raw = pd.Series([default_name] * len(tmp), index=tmp.index)
        else:
            raw = attack_df.loc[original_valid_index, col].fillna(default_name).astype(str)

        unique_values = sorted(raw.unique().tolist())
        mapping = {v: i + 1 for i, v in enumerate(unique_values)}
        encoded = raw.map(mapping).astype("int32")
        return raw, encoded, mapping

    stage_raw, stage_id, stage_mapping = encode_text_column(stage_col, "unknown_stage")
    type_raw, type_id, type_mapping = encode_text_column(type_col, "unknown_type")

    tmp["attack_stage_raw"] = stage_raw.values
    tmp["attack_stage_id"] = stage_id.values
    tmp["attack_type_raw"] = type_raw.values
    tmp["attack_type_id"] = type_id.values

    tmp = tmp.sort_values("start_sec").reset_index(drop=True)

    logger.info(f"成功加载攻击区间数量: {len(tmp)}")
    logger.info(f"攻击时间范围: {format_time_range(tmp['start_sec'].min(), tmp['end_sec'].max())}")

    with open("attack_label_mappings.json", "w", encoding="utf-8") as f:
        json.dump({
            "stage_mapping": stage_mapping,
            "type_mapping": type_mapping,
            "start_col": str(start_col),
            "end_col": str(end_col) if end_col else None,
            "duration_col": str(duration_col) if duration_col else None
        }, f, ensure_ascii=False, indent=2)

    return tmp


# =====================================================
# 5. 分块处理网络数据
# =====================================================
def process_network_data():
    if not os.path.exists(FILE_NETWORK):
        logger.warning(f"未找到网络数据文件: {FILE_NETWORK}")
        return pd.DataFrame(), [], None, None

    logger.info("=" * 80)
    logger.info(f"开始分块处理网络数据: {FILE_NETWORK}")

    sum_parts = []
    count_parts = []
    record_count_parts = []

    numeric_cols = None
    ts_col = None

    net_min_sec = np.inf
    net_max_sec = -np.inf

    for chunk_id, chunk in enumerate(pd.read_csv(FILE_NETWORK, chunksize=CHUNK_SIZE, low_memory=False), start=1):
        logger.info(f"  -> 网络数据第 {chunk_id} 块，行数: {len(chunk)}")

        if ts_col is None:
            ts_col = find_first_existing_col(chunk.columns, NETWORK_TS_CANDIDATES)
            if ts_col is None:
                raise ValueError(f"网络数据中找不到时间戳列。当前列名为: {list(chunk.columns)}")
            logger.info(f"网络时间戳列: {ts_col}")

        seconds = to_epoch_seconds(chunk[ts_col], ts_col)
        valid_time = seconds.notna()

        if not valid_time.any():
            logger.warning(f"网络数据第 {chunk_id} 块没有有效时间戳，跳过。")
            continue

        chunk = chunk.loc[valid_time].copy()
        seconds = seconds.loc[valid_time]

        net_min_sec = min(net_min_sec, float(seconds.min()))
        net_max_sec = max(net_max_sec, float(seconds.max()))

        chunk["window_id"] = make_window_id(seconds)

        if numeric_cols is None:
            exclude = set(EXCLUDE_NETWORK_COLS)
            exclude.add(str(ts_col).lower().strip())
            numeric_cols = detect_numeric_columns(chunk, exclude)

            if "window_id" in numeric_cols:
                numeric_cols.remove("window_id")

            logger.info(f"检测到网络数值特征数量: {len(numeric_cols)}")
            logger.info(f"网络数值特征前 10 个: {numeric_cols[:10]}")

        if len(numeric_cols) == 0:
            raise ValueError("未检测到任何网络数值特征，请检查 CSV 列类型。")

        data = chunk[numeric_cols].apply(pd.to_numeric, errors="coerce")
        data = data.replace([np.inf, -np.inf], np.nan)

        tmp = pd.concat([chunk["window_id"], data], axis=1)
        g = tmp.groupby("window_id", sort=False)

        sum_df = g[numeric_cols].sum(min_count=1).fillna(0.0)
        count_df = g[numeric_cols].count()
        record_count_df = g.size().rename("net_record_count").to_frame()

        sum_parts.append(sum_df)
        count_parts.append(count_df)
        record_count_parts.append(record_count_df)

    if not sum_parts:
        logger.warning("网络数据没有产生任何有效窗口。")
        return pd.DataFrame(), [], None, None

    logger.info("正在执行网络数据全局 sum/count 聚合...")

    total_sum = concat_groupby_sum(sum_parts)
    total_count = concat_groupby_sum(count_parts)
    total_record_count = concat_groupby_sum(record_count_parts)

    net_mean = total_sum.div(total_count.replace(0, np.nan)).fillna(0.0)
    net_agg = pd.concat([net_mean, total_record_count], axis=1).fillna(0.0)

    net_agg = net_agg.reset_index()
    net_agg["window_id"] = net_agg["window_id"].astype("int64")

    network_feature_cols = numeric_cols + ["net_record_count"]

    logger.info(f"网络数据处理完成，窗口数量: {len(net_agg)}")
    logger.info(f"网络时间范围: {format_time_range(net_min_sec, net_max_sec)}")
    logger.info("=" * 80)

    return net_agg, network_feature_cols, net_min_sec, net_max_sec


# =====================================================
# 6. 分块处理 provenance 数据
# =====================================================
def detect_prov_text_columns(chunk, time_cols):
    selected = []
    time_cols_lower = {str(c).lower().strip() for c in time_cols}

    for c in chunk.columns:
        c_lower = str(c).lower().strip()
        if c_lower in time_cols_lower:
            continue

        if not (pd.api.types.is_object_dtype(chunk[c]) or pd.api.types.is_string_dtype(chunk[c])):
            continue

        if any(h in c_lower for h in PROV_TEXT_HINTS):
            selected.append(c)

    return selected[:MAX_PROV_TEXT_COLS]


def detect_prov_unique_columns(chunk, time_cols):
    selected = []
    time_cols_lower = {str(c).lower().strip() for c in time_cols}

    for c in chunk.columns:
        c_lower = str(c).lower().strip()
        if c_lower in time_cols_lower:
            continue

        if not any(h in c_lower for h in PROV_UNIQUE_HINTS):
            continue

        sample = chunk[c].dropna().head(5000)
        if len(sample) == 0:
            continue

        if sample.nunique(dropna=True) > 1:
            selected.append(c)

        if len(selected) >= PROV_NUNIQUE_MAX_COLS:
            break

    return selected


def process_provenance_data():
    if not os.path.exists(FILE_PROVENANCE):
        logger.warning(f"未找到溯源数据文件: {FILE_PROVENANCE}")
        return pd.DataFrame(), [], None, None

    logger.info("=" * 80)
    logger.info(f"开始分块处理溯源数据: {FILE_PROVENANCE}")

    count_parts = []
    num_sum_parts = []
    num_count_parts = []

    unique_sets = defaultdict(lambda: defaultdict(set))

    time_cols = None
    text_cols = None
    unique_cols = None
    prov_numeric_cols = None

    prov_min_sec = np.inf
    prov_max_sec = -np.inf

    for chunk_id, chunk in enumerate(pd.read_csv(FILE_PROVENANCE, chunksize=CHUNK_SIZE, low_memory=False), start=1):
        logger.info(f"  -> 溯源数据第 {chunk_id} 块，行数: {len(chunk)}")

        if time_cols is None:
            time_cols = []
            for cand in PROVENANCE_TIME_CANDIDATES:
                col = find_first_existing_col(chunk.columns, [cand])
                if col is not None and col not in time_cols:
                    time_cols.append(col)

            if len(time_cols) == 0:
                raise ValueError(f"溯源数据中找不到时间列。当前列名为: {list(chunk.columns)}")

            logger.info(f"溯源时间候选列: {time_cols}")

        event_seconds = pd.Series(np.nan, index=chunk.index, dtype="float64")
        for tc in time_cols:
            sec = to_epoch_seconds(chunk[tc], tc)
            event_seconds = event_seconds.combine_first(sec)

        valid_time = event_seconds.notna()
        if not valid_time.any():
            logger.warning(f"溯源数据第 {chunk_id} 块没有有效时间戳，跳过。")
            continue

        chunk = chunk.loc[valid_time].copy()
        event_seconds = event_seconds.loc[valid_time]

        prov_min_sec = min(prov_min_sec, float(event_seconds.min()))
        prov_max_sec = max(prov_max_sec, float(event_seconds.max()))

        chunk["window_id"] = make_window_id(event_seconds)

        if text_cols is None:
            text_cols = detect_prov_text_columns(chunk, time_cols)
            logger.info(f"用于 provenance 关键词统计的文本列: {text_cols}")

        if unique_cols is None:
            unique_cols = detect_prov_unique_columns(chunk, time_cols)
            logger.info(f"用于 provenance 唯一实体统计的列: {unique_cols}")

        if prov_numeric_cols is None:
            exclude = {str(c).lower().strip() for c in time_cols}
            exclude.add("window_id")
            prov_numeric_cols = detect_numeric_columns(chunk, exclude)

            if "window_id" in prov_numeric_cols:
                prov_numeric_cols.remove("window_id")

            logger.info(f"检测到 provenance 数值列数量: {len(prov_numeric_cols)}")
            logger.info(f"provenance 数值列前 10 个: {prov_numeric_cols[:10]}")

        # 6.1 基础事件计数
        g = chunk.groupby("window_id", sort=False)
        count_df = g.size().rename("prov_event_count").to_frame()

        # 6.2 关键词事件计数
        if text_cols:
            for feat_name, pattern in KEYWORD_FEATURES.items():
                mask = pd.Series(False, index=chunk.index)

                for col in text_cols:
                    col_text = chunk[col].astype("string").str.lower()
                    mask = mask | col_text.str.contains(pattern, regex=True, na=False)

                tmp = pd.DataFrame({
                    "window_id": chunk["window_id"].values,
                    feat_name: mask.astype("int8").values
                })

                kw_count = tmp.groupby("window_id", sort=False)[feat_name].sum()
                count_df = count_df.join(kw_count, how="outer")

        count_df = count_df.fillna(0.0)
        count_parts.append(count_df)

        # 6.3 provenance 数值列均值，使用 sum/count 避免 mean-of-mean 偏差
        if prov_numeric_cols:
            num_data = chunk[prov_numeric_cols].apply(pd.to_numeric, errors="coerce")
            num_data = num_data.replace([np.inf, -np.inf], np.nan)

            num_tmp = pd.concat([chunk["window_id"], num_data], axis=1)
            ng = num_tmp.groupby("window_id", sort=False)

            nsum = ng[prov_numeric_cols].sum(min_count=1).fillna(0.0)
            ncnt = ng[prov_numeric_cols].count()

            num_sum_parts.append(nsum)
            num_count_parts.append(ncnt)

        # 6.4 精确唯一值统计
        if ENABLE_EXACT_PROV_NUNIQUE and unique_cols:
            for col in unique_cols:
                feat = f"prov_unique__{safe_name(col)}"
                grouped = chunk.groupby("window_id", sort=False)[col].agg(
                    lambda x: set(x.dropna().astype(str))
                )

                for wid, val_set in grouped.items():
                    unique_sets[feat][int(wid)].update(val_set)

    if not count_parts:
        logger.warning("溯源数据没有产生任何有效窗口。")
        return pd.DataFrame(), [], None, None

    logger.info("正在执行溯源数据全局聚合...")

    prov_count_all = concat_groupby_sum(count_parts)

    result_parts = [prov_count_all]

    prov_feature_cols = list(prov_count_all.columns)

    if num_sum_parts and num_count_parts:
        prov_num_sum = concat_groupby_sum(num_sum_parts)
        prov_num_count = concat_groupby_sum(num_count_parts)
        prov_num_mean = prov_num_sum.div(prov_num_count.replace(0, np.nan)).fillna(0.0)

        rename_map = {
            c: f"prov_num_mean__{safe_name(c)}"
            for c in prov_num_mean.columns
        }
        prov_num_mean = prov_num_mean.rename(columns=rename_map)

        result_parts.append(prov_num_mean)
        prov_feature_cols.extend(list(prov_num_mean.columns))

    if ENABLE_EXACT_PROV_NUNIQUE and unique_sets:
        all_wids = set()
        for feat, wid_dict in unique_sets.items():
            all_wids.update(wid_dict.keys())

        unique_df = pd.DataFrame(index=sorted(all_wids))

        for feat, wid_dict in unique_sets.items():
            unique_df[feat] = 0
            for wid, val_set in wid_dict.items():
                unique_df.loc[wid, feat] = len(val_set)

        result_parts.append(unique_df)
        prov_feature_cols.extend(list(unique_df.columns))

    prov_agg = pd.concat(result_parts, axis=1).fillna(0.0)
    prov_agg.index.name = "window_id"
    prov_agg = prov_agg.reset_index()
    prov_agg["window_id"] = prov_agg["window_id"].astype("int64")

    logger.info(f"溯源数据处理完成，窗口数量: {len(prov_agg)}")
    logger.info(f"溯源特征数量: {len(prov_feature_cols)}")
    logger.info(f"溯源时间范围: {format_time_range(prov_min_sec, prov_max_sec)}")
    logger.info("=" * 80)

    return prov_agg, prov_feature_cols, prov_min_sec, prov_max_sec


# =====================================================
# 7. 攻击标签对齐
# =====================================================
def attach_attack_labels(merged: pd.DataFrame, intervals: pd.DataFrame):
    window_start = merged["window_id"].values.astype("float64") * TIME_WINDOW_SEC
    window_end = window_start + TIME_WINDOW_SEC

    starts = np.sort(intervals["start_sec"].values.astype("float64"))
    ends = np.sort(intervals["end_sec"].values.astype("float64"))

    n_started = np.searchsorted(starts, window_end, side="left")
    n_ended = np.searchsorted(ends, window_start, side="right")

    attack_count = n_started - n_ended
    is_attack = (attack_count > 0).astype("float32")

    stage_id = np.zeros(len(merged), dtype="int32")
    type_id = np.zeros(len(merged), dtype="int32")

    for row in intervals.itertuples(index=False):
        lo = np.searchsorted(window_end, float(row.start_sec), side="right")
        hi = np.searchsorted(window_start, float(row.end_sec), side="left")

        if hi > lo:
            stage_id[lo:hi] = np.maximum(stage_id[lo:hi], int(row.attack_stage_id))
            type_id[lo:hi] = np.maximum(type_id[lo:hi], int(row.attack_type_id))

    merged["attack_count"] = attack_count.astype("float32")
    merged["is_attack"] = is_attack
    merged["attack_stage_id"] = stage_id
    merged["attack_type_id"] = type_id

    return merged


# =====================================================
# 8. 验证与保存
# =====================================================
def validate_and_save(
    merged,
    raw_matrix,
    obs_matrix,
    label_matrix,
    feature_cols,
    label_cols,
    intervals,
    net_range,
    prov_range
):
    logger.info("=" * 80)
    logger.info("开始执行数据质量验证...")

    is_attack = merged["is_attack"].values.astype(bool)

    nan_count = int(np.isnan(obs_matrix).sum())
    inf_count = int(np.isinf(obs_matrix).sum())
    nonzero_ratio = float(np.count_nonzero(obs_matrix) / max(obs_matrix.size, 1))
    attack_windows = int(is_attack.sum())
    total_windows = int(len(merged))
    attack_ratio = float(attack_windows / max(total_windows, 1))

    logger.info(f"特征矩阵 shape: {obs_matrix.shape}")
    logger.info(f"标签矩阵 shape: {label_matrix.shape}")
    logger.info(f"NaN 数量: {nan_count}")
    logger.info(f"Inf 数量: {inf_count}")
    logger.info(f"非零元素比例: {nonzero_ratio:.6f}")
    logger.info(f"攻击窗口数量: {attack_windows}")
    logger.info(f"总窗口数量: {total_windows}")
    logger.info(f"攻击窗口比例: {attack_ratio:.6f}")

    if attack_windows == 0:
        logger.warning("未匹配到任何攻击窗口。请重点检查 attack_info 与数据文件时间戳单位是否一致。")

    # 保存列名
    pd.Series(feature_cols, name="feature_name").to_csv("cicapt_feature_columns.csv", index=False)
    pd.Series(label_cols, name="label_name").to_csv("cicapt_label_columns.csv", index=False)

    # 保存完整窗口索引
    window_index_cols = [
        "window_id",
        "window_ts",
        "window_datetime_utc",
        "is_attack",
        "attack_count",
        "attack_stage_id",
        "attack_type_id"
    ]

    merged[window_index_cols].to_csv("cicapt_window_index.csv", index=False)

    # 保存攻击窗口
    attack_windows_df = merged.loc[merged["is_attack"] == 1.0, window_index_cols].copy()
    attack_windows_df.to_csv("verify_attack_windows.csv", index=False)

    if len(attack_windows_df) > 0:
        logger.info("前 5 个攻击窗口如下：")
        for _, row in attack_windows_df.head(5).iterrows():
            logger.info(
                f"  window_id={int(row['window_id'])}, "
                f"window_ts={row['window_ts']:.3f}, "
                f"datetime={row['window_datetime_utc']}, "
                f"attack_count={row['attack_count']}, "
                f"stage={row['attack_stage_id']}, "
                f"type={row['attack_type_id']}"
            )

    # 保存攻击窗口附近上下文
    if attack_windows > 0:
        attack_indices = np.where(is_attack)[0]
        selected_indices = set()

        for idx in attack_indices[:10]:
            lo = max(0, idx - 3)
            hi = min(len(merged), idx + 4)
            selected_indices.update(range(lo, hi))

        selected_indices = sorted(selected_indices)
        neighborhood_df = merged.iloc[selected_indices][window_index_cols].copy()
        neighborhood_df.to_csv("verify_attack_neighborhood.csv", index=False)

    # 比较攻击窗口和正常窗口特征差异
    if attack_windows > 0 and attack_windows < total_windows:
        benign_matrix = raw_matrix[~is_attack]
        attack_matrix = raw_matrix[is_attack]

        benign_mean = benign_matrix.mean(axis=0)
        attack_mean = attack_matrix.mean(axis=0)
        all_std = raw_matrix.std(axis=0) + EPS

        score = np.abs(attack_mean - benign_mean) / all_std

        diff_df = pd.DataFrame({
            "feature": feature_cols,
            "benign_mean": benign_mean,
            "attack_mean": attack_mean,
            "abs_standardized_diff": score
        }).sort_values("abs_standardized_diff", ascending=False)

        diff_df.to_csv("top_feature_differences.csv", index=False)

        logger.info("攻击/正常差异最大的前 10 个特征：")
        for _, row in diff_df.head(10).iterrows():
            logger.info(
                f"  {row['feature']}: diff={row['abs_standardized_diff']:.4f}, "
                f"benign_mean={row['benign_mean']:.4f}, "
                f"attack_mean={row['attack_mean']:.4f}"
            )

    summary = {
        "time_window_sec": TIME_WINDOW_SEC,
        "chunk_size": CHUNK_SIZE,
        "apply_zscore": APPLY_ZSCORE,
        "feature_shape": list(obs_matrix.shape),
        "label_shape": list(label_matrix.shape),
        "feature_dim": int(obs_matrix.shape[1]),
        "total_windows": total_windows,
        "attack_windows": attack_windows,
        "attack_ratio": attack_ratio,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "nonzero_ratio": nonzero_ratio,
        "network_time_range": {
            "min": None if net_range[0] is None else float(net_range[0]),
            "max": None if net_range[1] is None else float(net_range[1]),
            "readable": format_time_range(net_range[0], net_range[1])
        },
        "provenance_time_range": {
            "min": None if prov_range[0] is None else float(prov_range[0]),
            "max": None if prov_range[1] is None else float(prov_range[1]),
            "readable": format_time_range(prov_range[0], prov_range[1])
        },
        "attack_time_range": {
            "min": float(intervals["start_sec"].min()),
            "max": float(intervals["end_sec"].max()),
            "readable": format_time_range(intervals["start_sec"].min(), intervals["end_sec"].max())
        }
    }

    with open("extraction_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("验证文件已保存：")
    logger.info("  cicapt_feature_columns.csv")
    logger.info("  cicapt_label_columns.csv")
    logger.info("  cicapt_window_index.csv")
    logger.info("  verify_attack_windows.csv")
    logger.info("  verify_attack_neighborhood.csv")
    logger.info("  top_feature_differences.csv")
    logger.info("  extraction_summary.json")
    logger.info("=" * 80)


# =====================================================
# 9. 主流程
# =====================================================
def process_data():
    logger.info("=" * 80)
    logger.info("开始处理 CICAPT-IIoT 数据集")
    logger.info(f"时间窗口: {TIME_WINDOW_SEC} 秒")
    logger.info(f"分块大小: {CHUNK_SIZE} 行")
    logger.info(f"是否生成连续时间轴: {CREATE_CONTINUOUS_TIMELINE}")
    logger.info(f"是否进行 z-score: {APPLY_ZSCORE}")
    logger.info("=" * 80)

    # Step 1: 攻击区间
    intervals = load_attack_intervals()

    # Step 2: 网络数据
    net_agg, net_feature_cols, net_min, net_max = process_network_data()

    # Step 3: 溯源数据
    prov_agg, prov_feature_cols, prov_min, prov_max = process_provenance_data()

    if net_agg.empty and prov_agg.empty:
        raise RuntimeError("网络数据和溯源数据均为空，无法生成 npy。")

    # Step 4: 多源融合
    logger.info("=" * 80)
    logger.info("开始多源时间窗口融合...")

    dfs = []
    if not net_agg.empty:
        dfs.append(net_agg)
    if not prov_agg.empty:
        dfs.append(prov_agg)

    merged = reduce(
        lambda left, right: pd.merge(left, right, on="window_id", how="outer"),
        dfs
    )

    merged["window_id"] = merged["window_id"].astype("int64")
    merged = merged.sort_values("window_id").reset_index(drop=True)

    # Step 4.1: 生成连续时间轴
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

    # Step 4.2: 填充缺失特征
    all_feature_cols = net_feature_cols + prov_feature_cols

    for c in all_feature_cols:
        if c not in merged.columns:
            merged[c] = 0.0

    merged[all_feature_cols] = merged[all_feature_cols].fillna(0.0)

    # Step 5: 攻击标签对齐
    merged = attach_attack_labels(merged, intervals)

    logger.info(f"融合后总窗口数量: {len(merged)}")
    logger.info(f"融合后特征数量: {len(all_feature_cols)}")
    logger.info(f"攻击窗口数量: {int(merged['is_attack'].sum())}")
    logger.info("=" * 80)

    # Step 6: 构建特征矩阵
    raw_matrix = merged[all_feature_cols].values.astype("float32")
    raw_matrix = np.nan_to_num(raw_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    np.save(OUTPUT_FEATURES_RAW, raw_matrix)

    if APPLY_ZSCORE:
        mean = raw_matrix.mean(axis=0, keepdims=True)
        std = raw_matrix.std(axis=0, keepdims=True)
        std[std < EPS] = 1.0

        obs_matrix = ((raw_matrix - mean) / std).astype("float32")

        np.savez(
            "cicapt_feature_scaler.npz",
            mean=mean.astype("float32"),
            std=std.astype("float32")
        )
    else:
        obs_matrix = raw_matrix.astype("float32")

    # Step 7: 构建标签矩阵
    label_cols = [
        "risk_proxy_0_10",
        "service_pressure_proxy_0_5",
        "is_attack",
        "attack_stage_id"
    ]

    label_matrix = np.zeros((len(merged), 4), dtype="float32")

    label_matrix[:, 2] = merged["is_attack"].values.astype("float32")
    label_matrix[:, 3] = merged["attack_stage_id"].values.astype("float32")

    # 保持与你原主方法的 4 维 label 结构兼容。
    # 这里的 risk/service_pressure 是基于攻击窗口构造的代理标签。
    label_matrix[:, 0] = merged["is_attack"].values.astype("float32") * 10.0
    label_matrix[:, 1] = merged["is_attack"].values.astype("float32") * 5.0

    # Step 8: 保存 npy
    np.save(OUTPUT_FEATURES, obs_matrix)
    np.save(OUTPUT_LABELS, label_matrix)

    # Step 9: 验证与保存辅助文件
    validate_and_save(
        merged=merged,
        raw_matrix=raw_matrix,
        obs_matrix=obs_matrix,
        label_matrix=label_matrix,
        feature_cols=all_feature_cols,
        label_cols=label_cols,
        intervals=intervals,
        net_range=(None if not np.isfinite(net_min) else net_min,
                   None if not np.isfinite(net_max) else net_max),
        prov_range=(None if not np.isfinite(prov_min) else prov_min,
                    None if not np.isfinite(prov_max) else prov_max)
    )

    logger.info("✅ 数据提取完成")
    logger.info(f"输出特征文件: {OUTPUT_FEATURES}")
    logger.info(f"输出原始特征文件: {OUTPUT_FEATURES_RAW}")
    logger.info(f"输出标签文件: {OUTPUT_LABELS}")
    logger.info(f"特征矩阵 Shape: {obs_matrix.shape}")
    logger.info(f"标签矩阵 Shape: {label_matrix.shape}")


if __name__ == "__main__":
    process_data()