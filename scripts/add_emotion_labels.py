# -*- coding: utf-8 -*-
"""
add_emotion_labels.py — 曲风 → 情感标签映射 + 特征分析
=====================================================

【这个脚本做什么】
GTZAN 数据集只有 10 个「曲风」标签，没有「情感」标签。
本项目要做情感分析，所以必须先把曲风"翻译"成情感。
脚本做三件事：
  1) 读取 features.csv（37 维特征 + genre 曲风列），按映射表新增
     emotion_label（中文情感名）和 emotion_code（0~5 数值编码）两列，
     保存为 features_with_emotion.csv。
  2) 打印每种情感的样本数量分布；若某类 <50 条给出警告。
  3) 做特征分析：按情感分组算均值/标准差；用 ANOVA F 值找出区分度
     Top5 特征；用随机森林训练一次，输出特征重要性排序表。

【为什么这样映射（详见需求文档）】
  活力 = disco+reggae   激昂 = rock+metal   平静 = classical+jazz
  欢快 = pop            忧郁 = blues+country 律动 = hiphop
（样本分布：活力200 / 激昂200 / 平静199 / 欢快100 / 忧郁200 / 律动100，
 全部 ≥100，无类别过少问题。）

【运行】
  python scripts/add_emotion_labels.py
"""

from __future__ import annotations

import os
import pandas as pd
import numpy as np
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# 配置区（改映射只动这里）
# ============================================================

# 曲风 → 情感 的映射字典
GENRE_TO_EMOTION: dict[str, str] = {
    "disco": "活力",
    "reggae": "活力",
    "rock": "激昂",
    "metal": "激昂",
    "classical": "平静",
    "jazz": "平静",
    "pop": "欢快",
    "blues": "忧郁",
    "country": "忧郁",
    "hiphop": "律动",
}

# 情感的固定顺序，同时决定数值编码 0~5
EMOTION_ORDER: list[str] = ["活力", "激昂", "平静", "欢快", "忧郁", "律动"]

# 路径
DATA_CSV: str = "./data/features.csv"
OUT_CSV: str = "./data/features_with_emotion.csv"
MEAN_CSV: str = "./data/emotion_feature_mean.csv"
STD_CSV: str = "./data/emotion_feature_std.csv"
IMPORTANCE_CSV: str = "./data/feature_importance.csv"

# 样本量过少阈值：低于该值模型学不好，给出警告
MIN_SAMPLES_WARN: int = 50


# ============================================================
# 步骤 1：映射并加列
# ============================================================
def add_emotion_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    根据 GENRE_TO_EMOTION 给 DataFrame 新增 emotion_label 与 emotion_code 两列。

    参数
    ----
    df : pd.DataFrame
        必须包含 'genre' 列（曲风名）。

    返回
    ----
    pd.DataFrame
        原表 + 两列：emotion_label(中文), emotion_code(0~5)。
    """
    # 把曲风翻译成情感名
    df["emotion_label"] = df["genre"].map(GENRE_TO_EMOTION)

    # 若有曲风没在映射表里，map 会得到 NaN —— 必须检查，否则后面编码会出错
    missing = df["emotion_label"].isna().sum()
    if missing > 0:
        bad = df.loc[df["emotion_label"].isna(), "genre"].unique().tolist()
        raise ValueError(f"以下曲风未在映射表中定义，请补充：{bad}")

    # 情感名 → 0~5 数值编码（用 EMOTION_ORDER 的索引，保证可复现）
    emotion_to_code = {name: i for i, name in enumerate(EMOTION_ORDER)}
    df["emotion_code"] = df["emotion_label"].map(emotion_to_code)

    return df


# ============================================================
# 步骤 2 + 3：分布检查 + 特征分析
# ============================================================
def analyze_and_report(df: pd.DataFrame, feature_cols: list[str]) -> None:
    """
    打印情感分布、做 ANOVA 与随机森林特征重要性分析，并导出 CSV。

    参数
    ----
    df : pd.DataFrame
        含 emotion_label / emotion_code 及 37 个特征列。
    feature_cols : list[str]
        用于建模的 37 个特征列名。
    """
    # ---- 2) 各情感样本量分布 ----
    print("\n" + "=" * 60)
    print("各情感标签样本数量分布")
    print("=" * 60)
    counts = df["emotion_label"].value_counts().reindex(EMOTION_ORDER)
    for emotion, n in counts.items():
        print(f"  {emotion:<6} (code={EMOTION_ORDER.index(emotion)}): {n} 条")
    print(f"\n  合计：{int(counts.sum())} 条，情感类别数：{len(counts)}")

    # 样本量过少警告
    low = counts[counts < MIN_SAMPLES_WARN]
    if len(low) > 0:
        print("\n  ⚠ 警告：以下情感样本量偏少（<{}），模型可能学不好：".format(MIN_SAMPLES_WARN))
        for emotion, n in low.items():
            print(f"     - {emotion}: 仅 {n} 条")
            print(f"       建议：① 收集更多该情感样本；② 与相近情感合并；③ 用 class_weight 加权。")
    else:
        print(f"\n  ✓ 所有情感样本量均 ≥ {MIN_SAMPLES_WARN}，分布均衡，无需特别处理。")

    # ---- 3) 按情感分组：均值 / 标准差 ----
    print("\n" + "=" * 60)
    print("按情感分组的特征均值 / 标准差（完整表已存 CSV）")
    print("=" * 60)
    group_mean = df.groupby("emotion_label")[feature_cols].mean()
    group_std = df.groupby("emotion_label")[feature_cols].std()

    # 导出完整表，方便你用 Excel 细看
    group_mean.to_csv(MEAN_CSV, encoding="utf-8-sig")
    group_std.to_csv(STD_CSV, encoding="utf-8-sig")

    # 终端只打印几个"最能体现情感差异"的代表性特征，避免刷屏
    key_feats = ["tempo", "rms", "spectral_centroid", "mfcc_1", "chroma_1"]
    print("\n关键特征按情感均值（tempo=BPM, rms=能量, centroid=明亮度）：")
    print(group_mean[key_feats].round(3).to_string())

    # ---- 3a) ANOVA F 值：找区分度最高的特征 ----
    # f_classif 对每个特征做单因素方差分析，F 值越大说明该特征在不同情感间差异越显著。
    X = df[feature_cols].values
    y = df["emotion_code"].values
    F_vals, p_vals = f_classif(X, y)
    anova_df = (
        pd.DataFrame({"feature": feature_cols, "F": F_vals, "p_value": p_vals})
        .sort_values("F", ascending=False)
        .reset_index(drop=True)
    )

    print("\n" + "=" * 60)
    print("ANOVA F 值排行（Top 10，F 越大 = 区分度越高）")
    print("=" * 60)
    print(anova_df.head(10).round(3).to_string(index=False))
    top5_anova = anova_df.head(5)["feature"].tolist()
    print(f"\n  >>> 区分度 Top5 特征（ANOVA）：{top5_anova}")

    # ---- 3b) 随机森林特征重要性排序 ----
    # 用你后续训练阶段相同的 RF 配置，训练一次拿 feature_importances_ 作为"重要性"参考。
    print("\n" + "=" * 60)
    print("随机森林特征重要性排序（Top 15）")
    print("=" * 60)
    rf = RandomForestClassifier(
        n_estimators=100,   # 100 棵树
        max_depth=10,       # 限制深度，防过拟合
        random_state=42,    # 固定种子，结果可复现
        n_jobs=-1,          # 用满所有 CPU 核心
    )
    rf.fit(X, y)
    imp_df = (
        pd.DataFrame({"feature": feature_cols, "importance": rf.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    print(imp_df.head(15).round(4).to_string(index=False))
    imp_df.to_csv(IMPORTANCE_CSV, encoding="utf-8-sig")
    print(f"\n  >>> 特征重要性 Top5：{imp_df.head(5)['feature'].tolist()}")
    print(f"\n  （完整重要性表已保存：{IMPORTANCE_CSV}）")


# ============================================================
# 主流程
# ============================================================
def main() -> int:
    if not os.path.isfile(DATA_CSV):
        print(f"✗ 找不到 {DATA_CSV}，请先运行 extract_features.py")
        return 1

    df = pd.read_csv(DATA_CSV)
    # 37 个特征列 = 除了 genre 之外的所有列（此时还没有 emotion 列）
    feature_cols = [c for c in df.columns if c != "genre"]

    # 步骤 1：映射 + 加列
    df = add_emotion_labels(df)

    # 保存带情感标签的表（utf-8-sig 让 Excel 中文不乱码）
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"✓ 已保存：{OUT_CSV}  shape={df.shape}")

    # 步骤 2 + 3：分布检查与特征分析
    analyze_and_report(df, feature_cols)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
