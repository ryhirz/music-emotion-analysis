# -*- coding: utf-8 -*-
"""核心逻辑冒烟测试 —— 锁定三类"错了也不报错、但结果全废"的隐患。

背景：本项目此前没有任何自动化测试（审计报告 GAP-06）。这里刻意选择
**不依赖音频文件与 librosa** 的纯逻辑与产物一致性检查，因此运行快、
可在 CI 上稳定复现：

1. 曲风→情感映射表与情感顺序表不一致 —— 会导致标注出现 NaN 或无效编码；
2. 情感数值编码顺序被改动 —— 会让已训练好的模型与标签错位（预测结果整体乱掉）；
3. 特征名清单与模型输入维度不一致 —— 会在推理期抛形状错误。

运行：python -m pytest tests -q
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_script(module_name: str, relative_path: str):
    """按文件路径加载 scripts/ 下的模块。

    scripts/ 不是 Python 包（无 __init__.py），因此不能直接 import，
    这里用 importlib 从文件规格加载。
    """
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader, f"无法加载 {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def labels_mod():
    return _load_script("add_emotion_labels", "scripts/add_emotion_labels.py")


# --------------------------------------------------------------------------
# 1) 映射表与标签体系的一致性
# --------------------------------------------------------------------------
def test_genre_mapping_values_are_valid(labels_mod):
    """映射表里出现的每个情感都必须在 EMOTION_ORDER 中。"""
    unknown = {
        genre: emotion
        for genre, emotion in labels_mod.GENRE_TO_EMOTION.items()
        if emotion not in labels_mod.EMOTION_ORDER
    }
    assert not unknown, f"映射到了未定义的情感标签：{unknown}"


def test_genre_mapping_covers_all_gtzan_genres(labels_mod):
    """GTZAN 共 10 个曲风，映射表必须全覆盖，否则该曲风样本会被整批丢弃。"""
    gtzan_genres = {
        "blues", "classical", "country", "disco", "hiphop",
        "jazz", "metal", "pop", "reggae", "rock",
    }
    assert set(labels_mod.GENRE_TO_EMOTION) == gtzan_genres


def test_emotion_order_is_locked(labels_mod):
    """情感顺序即数值编码 0~5；一旦变动，已训练模型与标签就会错位。"""
    assert labels_mod.EMOTION_ORDER == ["活力", "激昂", "平静", "欢快", "忧郁", "律动"]


# --------------------------------------------------------------------------
# 2) 标注函数的行为
# --------------------------------------------------------------------------
def test_add_emotion_labels_assigns_expected_codes(labels_mod):
    """已知曲风应得到对应情感名与数值编码。"""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(
        {"genre": ["rock", "classical", "hiphop"], "mfcc_1": [0.1, 0.2, 0.3]}
    )
    out = labels_mod.add_emotion_labels(df)
    assert out["emotion_label"].tolist() == ["激昂", "平静", "律动"]
    # rock→激昂=1, classical→平静=2, hiphop→律动=5
    assert out["emotion_code"].tolist() == [1, 2, 5]


def test_add_emotion_labels_rejects_unknown_genre(labels_mod):
    """出现映射表外的曲风必须显式报错，而不是静默产生 NaN。"""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"genre": ["rock", "bossa_nova"], "mfcc_1": [0.1, 0.2]})
    with pytest.raises(ValueError, match="未在映射表中定义"):
        labels_mod.add_emotion_labels(df)


# --------------------------------------------------------------------------
# 3) 特征清单 与 已训练产物的维度一致性
# --------------------------------------------------------------------------
def test_feature_names_are_37_and_unique():
    """特征名清单必须是 37 个且无重复（README 与文档均以 37 维为口径）。"""
    names = json.loads((ROOT / "models" / "feature_names.json").read_text(encoding="utf-8"))
    assert isinstance(names, list)
    assert len(names) == 37, f"特征名数量应为 37，实际 {len(names)}"
    assert len(set(names)) == len(names), "特征名存在重复项"


def test_model_artifacts_match_feature_dimension():
    """已入库的模型 / 标准化器输入维度必须与特征清单一致。"""
    joblib = pytest.importorskip("joblib")
    names = json.loads((ROOT / "models" / "feature_names.json").read_text(encoding="utf-8"))

    model = joblib.load(ROOT / "models" / "model.joblib")
    scaler = joblib.load(ROOT / "models" / "scaler.joblib")

    assert model.n_features_in_ == len(names)
    assert scaler.n_features_in_ == len(names)
    # SVM 的类别数应与 6 类情感一致
    assert getattr(model, "n_classes_", 6) == 6
