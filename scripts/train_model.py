# -*- coding: utf-8 -*-
"""
train_model.py — 阶段 1-5：模型训练与评估
==========================================

【这个脚本做什么】
用 features_with_emotion.csv（37 维特征 + emotion_code 情感标签）训练 3 个模型，
对比它们的效果，挑出最好的一个保存，并做交叉验证。

【三个模型】
  - 随机森林 RandomForest：n_estimators=100, random_state=42
  - 支持向量机 SVM     ：kernel='rbf', C=1.0, random_state=42
  - K近邻 KNN         ：n_neighbors=5

【评估指标】（不止看准确率！）
  对每个模型输出：Accuracy、Precision(macro)、Recall(macro)、F1(macro)；
  画出混淆矩阵（seaborn.heatmap，存 PNG）；打印 classification_report；
  最后横向对比三模型。

【产出】
  - models/confusion_matrix_*.png         ：每个模型的混淆矩阵图
  - models/model.joblib                   ：F1 最高的模型
  - models/scaler.joblib                  ：训练时拟合的 StandardScaler
  - models/feature_names.json             ：37 个特征列名（保证预测时顺序一致）

【运行】
  python scripts/train_model.py
"""

from __future__ import annotations

import os
import json
import joblib

import numpy as np
import pandas as pd

# 无界面环境（服务器/后台）下保存图片必须指定 Agg 后端，否则plt.savefig会报错
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# 让图表支持中文：matplotlib 默认字体不含中文字形，不设的话坐标轴的
# 中文情感名会显示成方框（豆腐块）。Windows 上用 SimHei（黑体），
# 没有则回退到其他中文字体；最后用 Arial Unicode MS 兜底。
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False  # 避免负号显示为方块

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# 配置区
# ============================================================
DATA_CSV: str = "./data/features_with_emotion.csv"
MODEL_DIR: str = "./models"

# 情感标签顺序（与 emotion_code 0~5 对应），用于混淆矩阵坐标轴与分类报告
EMOTION_ORDER: list[str] = ["活力", "激昂", "平静", "欢快", "忧郁", "律动"]

# 不参与建模的列（genre=曲风, emotion_label=中文情感, emotion_code=数值标签）
NON_FEATURE_COLS: tuple[str, ...] = ("genre", "emotion_label", "emotion_code")


# ============================================================
# 1. 数据预处理
# ============================================================
def load_and_prepare() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    读取 CSV，切分 X（特征）和 y（情感编码），返回 (X, y, 特征列名)。
    """
    df = pd.read_csv(DATA_CSV)
    # 37 个特征列 = 除掉 3 个非特征列以外的所有列（动态获取，不怕以后增删特征）
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols].values          # 形状 (样本数, 37)
    y = df["emotion_code"].values        # 形状 (样本数,)，取值 0~5
    print(f"数据加载完成：{X.shape[0]} 条样本，{X.shape[1]} 个特征，{len(set(y))} 类情感")
    return X, y, feature_cols


# ============================================================
# 2 & 3. 训练 + 评估
# ============================================================
def train_and_evaluate(X_train, X_test, y_train, y_test) -> dict:
    """
    训练 3 个模型并逐个评估，返回每个模型的指标字典。
    同时把混淆矩阵画成 PNG 存到 MODEL_DIR。
    """
    # 三个模型的配置（严格按你的要求）
    models = {
        "随机森林": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        # ⚠ 必须加 probability=True：否则保存的模型没有 predict_proba()，
        #    阶段 1-6 的"6 类概率柱状图"就无法输出真实概率（只能做决策面软最大化近似）。
        #    probability=True 会让 SVM 在内部用 5 折交叉验证估计概率，预测结果不变。
        "SVM": SVC(kernel="rbf", C=1.0, random_state=42, probability=True),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }

    results: dict[str, dict] = {}

    for name, model in models.items():
        print("\n" + "=" * 64)
        print(f"训练并评估模型：{name}")
        print("=" * 64)

        # 训练（X_train 已经是标准化后的）
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # 四项指标（precision/recall/f1 都用 macro 平均，避免被大类掩盖小类表现）
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

        # 控制台打印分类报告（每类的 precision/recall/f1/support）
        print(f"\n分类报告（{name}）：")
        print(classification_report(y_test, y_pred, target_names=EMOTION_ORDER, digits=3, zero_division=0))

        # 混淆矩阵图
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6.2, 5.2))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=EMOTION_ORDER, yticklabels=EMOTION_ORDER,
        )
        plt.title(f"{name} 混淆矩阵", fontsize=13)
        plt.xlabel("预测情感", fontsize=11)
        plt.ylabel("真实情感", fontsize=11)
        plt.tight_layout()
        save_path = os.path.join(MODEL_DIR, f"confusion_matrix_{name}.png")
        plt.savefig(save_path, dpi=120)
        plt.close()
        print(f"  ✓ 混淆矩阵已保存：{save_path}")

    return results


def print_comparison(results: dict) -> str:
    """打印三模型指标横向对比表，返回最优模型名（按 F1）。"""
    cmp_df = pd.DataFrame(results).T  # 行=模型，列=指标
    print("\n" + "=" * 64)
    print("三模型指标横向对比（macro 平均）")
    print("=" * 64)
    print(cmp_df.round(4).to_string())

    best = max(results, key=lambda k: results[k]["f1"])
    print(f"\n>>> 按 F1 分数最高的模型是：{best} （F1={results[best]['f1']:.4f}）")
    return best


# ============================================================
# 主流程
# ============================================================
def main() -> int:
    if not os.path.isfile(DATA_CSV):
        print(f"✗ 找不到 {DATA_CSV}，请先运行 add_emotion_labels.py")
        return 1

    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1) 读取数据
    X, y, feature_cols = load_and_prepare()

    # 2) 划分训练集/测试集（8:2，分层抽样保证各类比例一致）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 3) 标准化：StandardScaler 只在训练集上 fit，再 transform 训练/测试集
    #    （测试集不能参与 fit，否则会数据泄露，评估就失真了）
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 4) 训练 + 评估
    results = train_and_evaluate(X_train_s, X_test_s, y_train, y_test)

    # 5) 对比 + 选最优
    best_name = print_comparison(results)

    # 6) 保存上线模型：为部署更稳健，用「全部数据（训练集+测试集）」重新标准化并拟合，
    #    让上线模型见过所有标注样本；对应的 scaler 也用全量数据 fit，二者必须配套。
    #    （训练/评估阶段用的是 train 子集，仅用于公平对比三模型；部署模型用全量更合理。）
    scaler_full = StandardScaler().fit(X)
    X_all_s = scaler_full.transform(X)
    best_model = {
        "随机森林": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "SVM": SVC(kernel="rbf", C=1.0, random_state=42, probability=True),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }[best_name]
    best_model.fit(X_all_s, y)

    joblib.dump(best_model, os.path.join(MODEL_DIR, "model.joblib"))
    joblib.dump(scaler_full, os.path.join(MODEL_DIR, "scaler.joblib"))
    with open(os.path.join(MODEL_DIR, "feature_names.json"), "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)
    print("\n✓ 已保存：model.joblib / scaler.joblib / feature_names.json（均已用全量数据 fit）")

    # 7) 对最优模型做 5 折交叉验证（用训练集，分层抽样）
    print(f"\n{best_name} 5 折交叉验证（在训练集上）：")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(best_model, X_train_s, y_train, cv=skf, scoring="accuracy", n_jobs=-1)
    print("  每折准确率：", np.round(cv_scores, 4).tolist())
    print(f"  均值 ± 标准差：{cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
