# -*- coding: utf-8 -*-
"""
predict.py — 阶段 1-6：单文件情感预测核心模块
=============================================

【这个脚本做什么】
给一段音频（wav / mp3），自动完成：
    1) 用 extract_features.py 里的 extract_features() 提取 37 维特征
    2) 用训练时保存的 StandardScaler 标准化
    3) 用保存的模型（当前是最优的 SVM）预测情感标签 + 6 类概率分布
返回：
    {
        "predicted_emotion": "活力",   # 预测的中文情感标签
        "confidence": 0.83,            # Top-1 置信度（0~1）
        "probabilities": {             # 6 类概率分布
            "活力": 0.83, "激昂": 0.05, ...
        }
    }

【异常处理（调用方拿到的是标准异常，方便界面层友好提示）】
    - 文件不存在            → FileNotFoundError
    - 格式不支持（非 wav/mp3）→ ValueError（提示仅支持 .wav / .mp3）
    - 音频加载失败（损坏）  → RuntimeError（提示文件可能损坏或非音频）
    - 特征维度不匹配        → ValueError（提示模型版本与特征提取器不兼容）

【为什么 mp3 也能用】
    librosa 读 mp3 依赖 audioread → ffmpeg。本机若没装系统 ffmpeg，
    这里会在导入时自动把 imageio-ffmpeg 自带的静态 ffmpeg 加进 PATH，
    从而零配置支持 mp3（不影响 wav）。若两者都没有，则只支持 wav。
"""

from __future__ import annotations

import os
import sys
import json
from typing import Dict, Optional

import numpy as np
import joblib
import librosa

# ============================================================
# mp3 解码支持：把 imageio-ffmpeg 自带的 ffmpeg 注入 PATH
# ============================================================
# audioread 在解码 mp3 时会去 PATH 里找 ffmpeg/avconv。
# 若用户没装系统级 ffmpeg，用 imageio-ffmpeg 的便携二进制顶上，做到"装好即用"。
try:
    import imageio_ffmpeg  # 轻量 pip 包，内嵌静态 ffmpeg 二进制
    _FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
    _FFMPEG_DIR = os.path.dirname(_FFMPEG_EXE)
    if _FFMPEG_DIR and _FFMPEG_DIR not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")
except Exception:
    # 没装 imageio-ffmpeg 也能跑（此时仅支持 wav，mp3 会走到 RuntimeError 提示）
    pass


# ============================================================
# 路径与配置
# ============================================================
# 本文件在 scripts/ 下；模型在 ../models/ 下
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# ⚠ 情感标签顺序：必须与训练脚本 train_model.py 的 EMOTION_ORDER 完全一致，
#    因为模型预测返回的是 0~5 的数值编码，靠这个列表映射回中文标签。
#    顺序错了，所有预测都会"张冠李戴"。
EMOTION_ORDER: list[str] = ["活力", "激昂", "平静", "欢快", "忧郁", "律动"]

# 支持的上传格式
SUPPORTED_EXTENSIONS = (".wav", ".mp3")

# 复用 extract_features.py 里的单文件特征提取函数（保持特征定义单一来源，避免两份不一致）
from extract_features import extract_features


# ============================================================
# 模型/标准化器/特征名：懒加载 + 缓存（多次预测只加载一次）
# ============================================================
_artifacts: Optional[tuple] = None  # (feature_names:list, model, scaler)


def _load_artifacts() -> tuple:
    """
    加载并缓存模型相关文件。第一次调用时从磁盘读取，之后复用内存中的实例。

    返回
    ----
    tuple : (feature_names, model, scaler)
        - feature_names : 37 个特征列名（训练时的顺序）
        - model         : 训练好的分类模型（当前是 SVM，带 predict_proba）
        - scaler        : 训练时拟合的 StandardScaler

    异常
    ----
    FileNotFoundError : 任意一个模型文件缺失（通常意味着还没训练）
    """
    global _artifacts
    if _artifacts is not None:
        return _artifacts

    feat_path = os.path.join(MODELS_DIR, "feature_names.json")
    model_path = os.path.join(MODELS_DIR, "model.joblib")
    scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")

    for path in (feat_path, model_path, scaler_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"找不到模型文件：{path}\n"
                f"（请先运行 scripts/train_model.py 训练并保存模型，"
                f"或确认 models/ 目录下有 model.joblib / scaler.joblib / feature_names.json）"
            )

    with open(feat_path, "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    _artifacts = (feature_names, model, scaler)
    return _artifacts


# ============================================================
# 核心函数：单文件情感预测
# ============================================================
def predict_emotion(audio_path: str) -> Dict[str, object]:
    """
    给定一段音频路径，返回预测的情感标签、置信度与 6 类概率分布。

    参数
    ----
    audio_path : str
        音频文件的完整路径（支持 .wav / .mp3）。

    返回
    ----
    Dict[str, object]，固定包含三个键：
        - "predicted_emotion" (str)  ：预测的中文情感标签
        - "confidence"        (float)：Top-1 类别的预测概率（0~1）
        - "probabilities"     (dict) ：键为中文情感名、值为概率（6 类，和为 1）

    异常
    ----
    FileNotFoundError : 文件路径不存在
    ValueError         : ① 格式不支持（非 wav/mp3）；② 特征维度与模型不匹配
    RuntimeError       : 音频加载失败（文件损坏、过短或非音频）
    """
    # ---- 1) 输入校验：文件是否存在 ----
    if not isinstance(audio_path, str) or not os.path.isfile(audio_path):
        raise FileNotFoundError(f"音频文件不存在：{audio_path}")

    # ---- 2) 输入校验：格式是否支持 ----
    ext = os.path.splitext(audio_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"不支持的文件格式：{ext or '无扩展名'}（仅支持 {', '.join(SUPPORTED_EXTENSIONS)}）"
        )

    # ---- 3) 特征提取（内部已 try/except，失败返回 None）----
    feat_dict = extract_features(audio_path)
    if feat_dict is None:
        # extract_features 返回 None 代表 librosa 无法解码/文件为空/损坏
        raise RuntimeError(
            "音频加载失败：文件可能已损坏、时长过短，或并非有效的音频文件。"
        )

    # ---- 4) 加载模型并严格按训练顺序组装特征向量 ----
    feature_names, model, scaler = _load_artifacts()

    try:
        # 按 feature_names.json 的顺序取特征，保证与训练时输入列顺序一致
        vec = np.array(
            [float(feat_dict[name]) for name in feature_names],
            dtype=float,
        )
    except KeyError as miss:
        raise ValueError(
            f"特征缺失，模型版本与特征提取器不兼容：缺少特征 {miss}。"
            f"请确认 extract_features.py 与训练时的特征定义一致。"
        )

    # ---- 5) 维度校验：防止模型/特征提取器版本错位 ----
    if vec.shape[0] != len(feature_names):
        raise ValueError(
            f"特征维度不匹配：实际得到 {vec.shape[0]} 维，"
            f"模型要求 {len(feature_names)} 维（模型版本与特征提取器不兼容）。"
        )

    # ---- 6) 标准化 + 预测 ----
    X = scaler.transform(vec.reshape(1, -1))          # 转成 (1, 37) 再标准化
    classes = model.classes_                          # 训练时的类别编码，如 [0,1,2,3,4,5]
    proba = model.predict_proba(X)[0]                 # 每类概率，长度 = 类别数

    # 组装 6 类概率分布（键用中文情感名）
    probabilities: Dict[str, float] = {
        EMOTION_ORDER[int(code)]: float(p) for code, p in zip(classes, proba)
    }

    # Top-1：概率最大的类
    top_idx = int(np.argmax(proba))
    predicted_code = int(classes[top_idx])
    predicted_emotion = EMOTION_ORDER[predicted_code]
    confidence = float(proba[top_idx])

    return {
        "predicted_emotion": predicted_emotion,
        "confidence": confidence,
        "probabilities": probabilities,
    }


# ============================================================
# 测试入口：直接 `python scripts/predict.py [音频路径]` 即可打印预测结果
# ============================================================
if __name__ == "__main__":
    # 允许从命令行传入音频路径；不传则用一个数据集里的样本（pop→欢快）演示
    if len(sys.argv) > 1:
        test_audio = sys.argv[1]
    else:
        test_audio = os.path.join(
            PROJECT_ROOT, "data", "genres_original", "pop", "pop.00000.wav"
        )

    try:
        result = predict_emotion(test_audio)
        print("预测结果：")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (FileNotFoundError, ValueError, RuntimeError) as err:
        # 这些都是"可预期的"业务异常，友好打印，退出码 1
        print(f"✗ 预测失败：{type(err).__name__}: {err}")
        sys.exit(1)
    except Exception as err:
        # 其余未预期异常，原样抛出便于排查
        print(f"✗ 预测失败（未预期）：{type(err).__name__}: {err}")
        sys.exit(2)
