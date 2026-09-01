# -*- coding: utf-8 -*-
"""
app.py — 阶段 1-6：Gradio 单文件情感预测界面
=============================================

【这个界面做什么】
用户上传一段音频（wav / mp3）→ 自动提取 37 维特征 → SVM 预测 →
展示 Top-1 中文情感标签 + 6 类概率分布柱状图。

【项目说明模块】
界面顶部内置「项目说明」，诚实标注 GTZAN 的"弱监督"局限：
GTZAN 只有曲风标签、没有真实情感标注，本项目用「曲风→情感」规则
映射构造标签，模型学到的主要是曲风声学差异。

【如何运行】
    # 1) 激活虚拟环境（Windows）
    .venv/Scripts/activate
    # 2) 启动界面
    python app.py
    然后浏览器打开 http://127.0.0.1:7860

【异常容错】
- 上传非 wav/mp3：提示仅支持这两种格式
- 文件损坏 / 非音频：提示可能已损坏，不会让服务崩溃
- 模型文件缺失：提示先运行 train_model.py
所有异常都被界面层捕获并友好展示。
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")  # 无界面（后台/服务器）环境下绘图必须指定 Agg
import matplotlib.pyplot as plt

import gradio as gr

# 把 scripts/ 加入模块搜索路径，才能 import predict（以及它内部的 extract_features）
SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from predict import predict_emotion, EMOTION_ORDER  # noqa: E402

# 让图表支持中文（同 train_model.py：Windows 用 SimHei/微软雅黑兜底）
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# Apple 极简风主色
ACCENT = "#007AFF"


# ============================================================
# 概率柱状图（6 类，按概率降序）
# ============================================================
def make_bar_chart(probabilities: dict) -> plt.Figure:
    """根据各情感概率，画一张带百分比标注的条形图。"""
    # 按概率从高到低排序，观感更清晰
    ordered = sorted(probabilities.items(), key=lambda x: -x[1])
    labels = [k for k, _ in ordered]
    vals = [v for _, v in ordered]

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    bars = ax.bar(labels, [v * 100 for v in vals], color=ACCENT, width=0.6)
    ax.set_ylabel("概率 (%)", fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_title("6 类情感概率分布", fontsize=12)
    # 在每根柱子上方标注百分比
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2, v * 100 + 1.5,
            f"{v * 100:.1f}%", ha="center", fontsize=9,
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


# ============================================================
# 核心回调：上传音频 → 预测 → 返回结果
# ============================================================
def analyze(audio_path: str | None):
    """
    Gradio 的按钮/上传回调。

    参数
    ----
    audio_path : str | None
        Gradio gr.Audio(type="filepath") 传来的本地文件路径；未上传为 None。

    返回
    ----
    (str, plt.Figure | None, str) ：
        - 预测结果 Markdown
        - 概率柱状图（失败为 None）
        - 详细概率 Markdown
    """
    # 1) 没上传就点预测
    if audio_path is None:
        return "请先上传一段音频（wav / mp3）。", None, ""

    try:
        result = predict_emotion(audio_path)
        label = result["predicted_emotion"]
        conf = result["confidence"]
        probs = result["probabilities"]

        # Top-1 结果卡片
        summary = (
            f"## 🎯 预测情感：**{label}**\n\n"
            f"**置信度**：{conf * 100:.1f}%\n\n"
            f"> 模型在 6 类情感上的判断分布见右图。"
        )

        # 概率柱状图
        fig = make_bar_chart(probs)

        # 详细概率列表（按概率降序）
        detail_lines = ["### 各情感概率明细", ""]
        for name, p in sorted(probs.items(), key=lambda x: -x[1]):
            detail_lines.append(f"- **{name}**：{p * 100:.1f}%")
        detail = "\n".join(detail_lines)

        return summary, fig, detail

    # 2) 可预期的"业务异常"：友好提示，不 crash 服务
    except (FileNotFoundError, ValueError, RuntimeError) as err:
        msg = (
            "## ⚠ 处理失败\n\n"
            f"{err}\n\n"
            "**排查建议**：\n"
            "- 仅支持 `.wav` / `.mp3` 格式；\n"
            "- 确认文件未损坏、是有效的音频；\n"
            "- 若提示模型文件缺失，请先运行 `python scripts/train_model.py`。"
        )
        return msg, None, ""
    # 3) 其他未预期异常：原样报出，便于定位
    except Exception as err:
        return f"## ⚠ 未知错误\n\n`{type(err).__name__}`: {err}", None, ""


# ============================================================
# 界面布局
# ============================================================
PROJECT_INFO = """
### 📁 数据集：GTZAN
- 1000 首、每首约 30 秒的音频片段，分为 **10 个曲风**（blues / classical / country / disco / hiphop / jazz / metal / pop / reggae / rock）。
- 这是音乐信息检索领域最常用的公开基准数据集之一。

### 🔗 曲风 → 情感 映射（本项目核心设计）
| 情感标签 | 对应曲风 | 依据（声学视角） |
|---|---|---|
| 活力 | disco, reggae | 中高 BPM、强律动、明亮度高 |
| 激昂 | rock, metal | 高能量、高失真、频谱宽（对比度大） |
| 平静 | classical, jazz | 低能量、柔和动态、节奏舒缓 |
| 欢快 | pop | 中快 BPM、大调明亮、旋律上扬 |
| 忧郁 | blues, country | 小调/抒情、低能量、速度偏慢 |
| 律动 | hiphop | 中速、强低频贝斯、节奏突出 |

### ⚠️ 诚实声明：弱监督局限性
> **GTZAN 只有"曲风"标签，没有真实的"情感"标注。** 本项目的情感标签是通过上述「曲风→情感」规则**人工映射**得到的，属于**弱监督**学习。
> 因此模型本质上学到的是"不同曲风的声学差异"，而非人类主观情感。这是该数据集做情感分析的固有天花板——
> **测试集 Macro-F1 在 60%~75% 属于正常区间**，过高（如 95%+）反而说明模型只是"记住了曲风"而非理解情感。
> 后续若要更贴近真实情感，应改用带原生情感标注的数据集（如 DEAM）重新训练。

### 🧩 技术方案
- 特征：37 维传统音频特征（MFCC / 频谱对比度 / 色度 / 能量 / 速度等），**未使用深度学习**。
- 模型：SVM（在 3 个候选模型中 Macro-F1 最高）。
- 特征重要性分析发现：**频谱对比度（Spectral Contrast）** 是区分情感的最强特征，而 **BPM 未进入 Top10**——因为 GTZAN 多数曲风节奏都挤在 110~130 BPM。
"""

with gr.Blocks(title="音乐情感分析工具") as demo:
    gr.Markdown("# 🎵 音乐情感分析工具\n上传一段音频，自动识别它最可能表达的情感（活力 / 激昂 / 平静 / 欢快 / 忧郁 / 律动）。")

    with gr.Accordion("📖 项目说明（点击展开）", open=False):
        gr.Markdown(PROJECT_INFO)

    with gr.Row():
        with gr.Column(scale=1):
            audio_in = gr.Audio(
                label="上传音频（支持 wav / mp3）",
                sources=["upload"],
                type="filepath",
            )
            btn = gr.Button("开始分析", variant="primary")
        with gr.Column(scale=1):
            result_md = gr.Markdown(label="预测结果")
            plot_out = gr.Plot(label="6 类概率分布")
            detail_md = gr.Markdown(label="详细概率")

    # 点击按钮或上传后回车都触发分析
    btn.click(
        fn=analyze,
        inputs=audio_in,
        outputs=[result_md, plot_out, detail_md],
    )
    audio_in.change(
        fn=analyze,
        inputs=audio_in,
        outputs=[result_md, plot_out, detail_md],
    )


# ============================================================
# 启动（仅直接运行本文件时）
# ============================================================
if __name__ == "__main__":
    # share=False：仅本机访问（不生成公网链接）；如需部署到 Hugging Face Spaces 再改 True
    # 部署适配：本地固定 127.0.0.1:7860；
    # Hugging Face Spaces 会注入 PORT 环境变量，并要求绑定 0.0.0.0 才能对外暴露端口。
    # 用同一段代码兼容两种环境（0.0.0.0 下本地访问 127.0.0.1:7860 同样可达）。
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
    )
