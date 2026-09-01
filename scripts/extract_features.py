# -*- coding: utf-8 -*-
"""
extract_features.py — GTZAN 音频特征提取脚本
=============================================

【这个脚本做什么】
遍历 data/genres_original/ 下每个曲风文件夹里的音频，
用 librosa 提取 37 维音频特征（每维取整首歌的均值），
再拼成一张大表（DataFrame）保存到 data/features.csv。
这张表就是你后面训练情感分类模型时的"食材"。

【特征维度一览（共 37 维 + 1 列 genre 标签）】
  - MFCC（梅尔频率倒谱系数）      ：13 维  mfcc_1 ~ mfcc_13
  - 频谱质心 spectral_centroid     ： 1 维
  - 频谱滚降点 spectral_rolloff    ： 1 维
  - 过零率 zero_crossing_rate      ： 1 维
  - RMS 能量 rms                   ： 1 维
  - 色度特征 chroma_stft           ：12 维  chroma_1 ~ chroma_12
  - 速度 tempo (BPM)               ： 1 维
  - 频谱对比度 spectral_contrast   ： 7 维  contrast_1 ~ contrast_7
  ----------------------------------------------------------
  合计 37 维

【为什么用这些特征】
它们覆盖了节奏（tempo）、音高/谐波（chroma、mfcc）、
响度/能量（rms）、音色明亮度（spectral_centroid/rolloff/contrast）、
以及时域粗糙度（zero_crossing_rate）。
对"曲风→情感"这类任务，这些传统特征已经足够，不需要深度学习。

【如何运行】
  python scripts/extract_features.py
运行完成后会打印 DataFrame 的 shape 和前 5 行，并保存 data/features.csv。
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm


# ============================================================
# 全局配置（改这里就能调整，不需要动下面的逻辑）
# ============================================================

# 统一采样率：22050 Hz 是 librosa 默认值，也正好是 GTZAN 原始采样率，
# 统一采样率能保证所有歌曲的特征在同一尺度上，模型才好学。
SR: int = 22050

# GTZAN 的 10 个曲风文件夹名（也是后面训练时的分类标签）。
# 注意：我们做的是"情感分析"，但 GTZAN 只有曲风标签，
# 后续需要在代码/映射表里把「曲风 → 情感」对应起来（见需求文档）。
GENRES: List[str] = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
]

# 数据根目录：脚本会读取 ./data/genres_original/，结果写到 ./data/features.csv
DATA_ROOT: str = "./data"
GENRES_DIR: str = os.path.join(DATA_ROOT, "genres_original")
OUTPUT_CSV: str = os.path.join(DATA_ROOT, "features.csv")


# ============================================================
# 核心函数：单文件特征提取
# ============================================================
def extract_features(file_path: str) -> Optional[Dict[str, float]]:
    """
    提取单首音频的 37 维特征（每维取整首歌的均值）。

    参数
    ----
    file_path : str
        音频文件的完整路径（.wav 或 .au 均可）。

    返回
    ----
    Optional[Dict[str, float]]
        - 成功：返回形如 {"mfcc_1": 0.12, "spectral_centroid": 1234.5, ...} 的字典，
          字典的 key 就是特征列名。
        - 失败（文件损坏/无法解码）：打印警告并返回 None，
          由调用方决定是否跳过，**不会让整个程序崩溃**。

    设计说明
    --------
    所有 librosa.feature.* 提取出来的都是二维数组，形状为 (维度数, 时间帧数)。
    例如 mfcc 是 (13, T)，表示 13 个系数各自在 T 个时间帧上的取值。
    我们对其取"按行均值"（.mean(axis=1)），就得到每个系数"整首歌的平均值"，
    再把每一维单独展开成 mfcc_1 ~ mfcc_13 这样的列。
    """
    try:
        # 1) 加载音频：统一重采样到 SR，并转成单声道（mono=True）。
        #    y 是波形数组（一维），sr 是采样率。
        y, sr = librosa.load(file_path, sr=SR, mono=True)

        # 若加载成功但音频为空（极少见），直接当作失败处理
        if y.size == 0:
            print(f"  ⚠ 警告：文件为空，已跳过 -> {file_path}")
            return None

        feat: Dict[str, float] = {}

        # 2) MFCC（梅尔频率倒谱系数）：描述音色的核心特征，13 维。
        #    n_mfcc=13 明确指定只要 13 维（librosa 默认是 20 维）。
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # 形状 (13, T)
        for i in range(mfcc.shape[0]):
            feat[f"mfcc_{i + 1}"] = float(mfcc[i].mean())

        # 3) 频谱质心：声音"明亮度"的度量（高频能量占比高 → 质心靠后）。
        #    返回 (1, T)，.mean() 直接得到整首歌的标量均值。
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        feat["spectral_centroid"] = float(cent.mean())

        # 4) 频谱滚降点：累计能量达到 85% 时对应的频率，反映音色"厚度"。
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        feat["spectral_rolloff"] = float(rolloff.mean())

        # 5) 过零率：波形穿过零轴的频率，粗粒度反映"粗糙度/清脆度"。
        zcr = librosa.feature.zero_crossing_rate(y=y)
        feat["zero_crossing_rate"] = float(zcr.mean())

        # 6) RMS 能量：响度的客观度量，能量高通常对应"活力/激昂"。
        rms = librosa.feature.rms(y=y)
        feat["rms"] = float(rms.mean())

        # 7) 色度特征（chroma_stft）：12 维，对应音乐里的 12 个半音类，
        #    能捕捉和声/旋律走向，对"欢快/悲伤"这类情感很有区分度。
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)  # 形状 (12, T)
        for i in range(chroma.shape[0]):
            feat[f"chroma_{i + 1}"] = float(chroma[i].mean())

        # 8) 速度 tempo（BPM）：乐曲快慢，是"活力 vs 平静"最强信号之一。
        #    librosa.feature.tempo 返回形状 (1,) 的数组，用 atleast_1d 兜底再取均值。
        tempo = librosa.feature.tempo(y=y, sr=sr)
        feat["tempo"] = float(np.atleast_1d(tempo).mean())

        # 9) 频谱对比度（spectral_contrast）：7 维。
        #    对比度大 = 音色"尖锐/有攻击性"（金属、摇滚）；小 = "柔和/平缓"。
        #    默认 n_bands=6，加上最低频带共 7 维，正好对应 contrast_1 ~ contrast_7。
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)  # 形状 (7, T)
        for i in range(contrast.shape[0]):
            feat[f"contrast_{i + 1}"] = float(contrast[i].mean())

        return feat

    except Exception as err:  # 捕获一切加载/解码异常，保证程序不中断
        print(f"  ⚠ 警告：特征提取失败，已跳过 -> {file_path}")
        print(f"      原因：{type(err).__name__}: {err}")
        return None


# ============================================================
# 批量处理主逻辑
# ============================================================
def main() -> int:
    """
    批量遍历所有曲风文件夹，提取特征并汇总成 DataFrame。

    返回
    ----
    int : 进程退出码（0 表示正常结束）。
    """
    # 防御性检查：数据目录必须存在
    if not os.path.isdir(GENRES_DIR):
        print(f"✗ 找不到数据集目录：{GENRES_DIR}")
        print("  请先运行 scripts/download_gtzan.py 下载数据集。")
        return 1

    rows: List[Dict[str, float]] = []   # 每行 = 一首歌的全部特征
    skipped: int = 0                     # 被跳过的损坏文件计数

    # 外层进度条：10 个曲风；内层进度条：每个曲风里的文件。
    # tqdm 就是那个绿色进度条，让你知道程序跑到哪了、还要多久。
    for genre in tqdm(GENRES, desc="曲风", unit="genre"):
        genre_dir = os.path.join(GENRES_DIR, genre)
        if not os.path.isdir(genre_dir):
            print(f"  ⚠ 跳过不存在的曲风目录：{genre}")
            continue

        # 只处理音频文件（过滤 macOS 垃圾文件 ._xxx 等）
        audio_files = sorted(
            f for f in os.listdir(genre_dir)
            if f.lower().endswith((".wav", ".au")) and not f.startswith("._")
        )

        for filename in tqdm(audio_files, desc=f"  {genre}", unit="file", leave=False):
            file_path = os.path.join(genre_dir, filename)
            feat = extract_features(file_path)
            if feat is None:
                skipped += 1
                continue
            # 把曲风标签作为一列加进字典，方便后面监督学习
            feat["genre"] = genre
            rows.append(feat)

    if not rows:
        print("✗ 没有成功提取到任何特征，请检查数据集是否完整。")
        return 1

    # 组装成 DataFrame：每一行一首歌，列 = 37 个特征 + genre
    df = pd.DataFrame(rows)

    # 固定列顺序：先 37 个特征，最后 genre（features.csv 的列顺序更清晰）
    feature_cols = [c for c in df.columns if c != "genre"]
    df = df[feature_cols + ["genre"]]

    # 保存为 CSV。utf-8-sig 编码让 Excel 直接打开中文/英文都不会乱码。
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n✓ 特征已保存至：{OUTPUT_CSV}")
    print(f"  成功提取 {len(df)} 首，跳过 {skipped} 首。")

    # 打印摘要，方便你一眼确认结果对不对
    print(f"\nDataFrame shape（行数=歌曲数, 列数={len(feature_cols)}特征+1标签）：{df.shape}")
    print("\n前 5 行预览：")
    # 列太多时横向会被截断，这里只显示前 8 个特征列 + genre 便于阅读
    preview_cols = feature_cols[:8] + ["genre"]
    print(df[preview_cols].head(5).to_string())

    return 0


# ============================================================
# 入口：只有直接运行这个脚本时才执行 main()
# （被别的脚本 import 时不会自动跑，方便你复用 extract_features 函数）
# ============================================================
if __name__ == "__main__":
    sys.exit(main())
