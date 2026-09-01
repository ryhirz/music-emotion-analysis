# 📦 代码托管与部署指南（Deployment Guide）

本指南帮助你把 **MusicEmotion AI** 整洁地上传到 GitHub，并免费部署到 Hugging Face Spaces，让面试官 / 开源社区能直接访问你的在线 Demo。

---

## 概览

| 目标 | 方式 | 费用 |
|------|------|------|
| 代码托管（整洁仓库） | GitHub + `.gitignore` | 免费 |
| 在线 Demo 部署 | Hugging Face Spaces（Gradio SDK, Free CPU） | 免费 |

---

## Part 1：规范上传到 GitHub

### 1.1 仓库整洁度：`.gitignore` 已就位

仓库根目录已提供 `.gitignore`，关键排除项：

- `.venv/` `__pycache__/` —— 虚拟环境与缓存，绝不入库；
- `*.mp3` `*.wav` `*.au` `*.flac` `*.ogg` —— 音频文件不入库；
- `data/genres_original/` `data/_tmp_extract/` —— GTZAN 原始音频约 **1.3GB**，请通过 `scripts/download_gtzan.py` 重新获取；
- `temp_*` `*.tmp` —— 临时文件。

**会被正常跟踪的文件**：`assets/*.png`（界面截图 / 混淆矩阵）、`data/*.csv`（特征表，体积小）、`models/*.joblib`（模型文件，见下）、全部 `scripts/*.py`、`app.py`、`requirements.txt`、`README.md`、`DEPLOYMENT.md`。

### 1.2 大文件处理警告（实测结论，避免踩坑）

> ⚠️ **GitHub 单文件硬上限 = 100MB**，超过会直接拒绝 `push`。

先检查你的模型文件大小：

```bash
ls -lh models/
# model.joblib     292K
# scaler.joblib    4.0K
# feature_names.json  1.0K
```

**本项目实测结论**：`model.joblib` 仅 **292KB**，**远低于 50MB / 100MB 限制**，可直接 `git push`，**无需 Git LFS、也无需上传到 Releases / 网盘**。这样别人 `clone` 后开箱即用，体验最好。

> **未来若模型变大（如换深度学习模型 > 50MB），两种替代方案：**
> - **方案 A · Git LFS**：`git lfs install` → `git lfs track "models/*.joblib"` → 正常 add/commit/push（LFS 有免费额度，超出需付费）；
> - **方案 B · Release / 网盘**：把模型放到 GitHub Releases 或网盘，README 提供下载链接，并在 `app.py` 启动前自动下载到 `models/`。

### 1.3 标准 Push 流程

```bash
# 在仓库根目录（已含 .gitignore）
git init
git add .
git commit -m "Initial commit: MusicEmotion AI — 音乐情感分析工具"

# 关联远程（把 <你的用户名> 换成实际 GitHub 用户名）
git branch -M main
git remote add origin git@github.com:<你的用户名>/MusicEmotion-AI.git
git push -u origin main
```

> `git add .` 会**自动遵守 `.gitignore`**，不会把 `.venv` 和 1.3GB 原始音频带进去。推送前可用 `git status` 复查待提交文件清单。

---

## Part 2：部署到 Hugging Face Spaces

### 2.1 创建 Space

1. 登录 [huggingface.co](https://huggingface.co)，右上角 **New Space**；
2. **SDK** 选择 **Gradio**；
3. **Hardware** 选择 **Free（CPU）tier**；
4. Space 名称填 `MusicEmotion-AI`，可见性选 `Public`（简历展示建议 Public）；
5. 创建后即可获得该 Space 的 Git 仓库地址。

### 2.2 代码适配（已完成 ✅）

`app.py` 末尾的 `demo.launch()` 已改为自适应两种环境：

```python
demo.launch(
    server_name="0.0.0.0",                       # HF 要求绑定 0.0.0.0 才能对外暴露
    server_port=int(os.environ.get("PORT", 7860)),  # 读取 HF 注入的 PORT 环境变量
    share=False,
)
```

- **本地**：访问 `http://127.0.0.1:7860`（与改动前一致）；
- **HF Spaces**：自动读取平台注入的 `PORT` 并绑定 `0.0.0.0`，对外正常暴露端口。

### 2.3 依赖与运行环境（最稳妥的「mp3 零报错」方案）

**结论先行：本项目在 HF Spaces 上不需要 `packages.txt`，也不需要 `Dockerfile`。**

原因——本项目读 `mp3` 依赖 `librosa` → `audioread` → `ffmpeg`，但你无需在 Space 上安装系统级 ffmpeg：

- `requirements.txt` 已包含 **`imageio-ffmpeg`**，它在安装时会自带一个**静态 ffmpeg 二进制**；
- `scripts/predict.py` 在运行时会调用 `imageio_ffmpeg.get_ffmpeg_exe()` 把该二进制路径注入 `PATH`；
- 因此 `librosa.load("xxx.mp3")` 在 HF Spaces 上**开箱即用、零额外配置**。

> （可选）如果你更想用系统级 ffmpeg，可在仓库根目录新建 `packages.txt` 写入一行 `ffmpeg`，HF 会在构建期安装系统 ffmpeg——但**对本项目非必需**，依赖 `imageio-ffmpeg` 已足够稳妥。

**Python 版本提示**：HF Spaces 默认 Python 3.10 / 3.11。本项目 `requirements.txt` 在该版本可正常安装（`numba==0.61` 支持 3.10+；`standard-aifc`/`standard-sunau` 在 3.13 才需要、低版本装了也无害）。建议在 Space 设置的 **Python version** 选 **3.11**（与本地实测最顺滑）。

### 2.4 上传 `requirements.txt` 与模型文件

随代码一起 `push` 的文件（均已被 `.gitignore` 正确放行）：

- `requirements.txt` —— HF 自动 `pip install -r requirements.txt`；
- `models/model.joblib`、`models/scaler.joblib`、`models/feature_names.json` —— 模型小，直接进 git（无需 LFS）；
- `app.py`、`scripts/`、`assets/`、`README.md`、`DEPLOYMENT.md`。

### 2.5 推送命令

```bash
# 在本地仓库根目录（已含全部需部署文件）
git init
git add .
git commit -m "Deploy MusicEmotion AI to HF Spaces"

git branch -M main
git remote add space https://huggingface.co/spaces/<你的用户名>/MusicEmotion-AI
git push -u space main
```

> 若已关联过其它 remote，可用 `git remote set-url space <地址>` 或换名。推送后 HF 自动构建，约 1–2 分钟就绪，访问 Space 网址即可使用。

### 2.6 部署后验证

1. 打开 Space 网址，上传一首 `pop` / `rock` 音频，确认返回中文情感标签 + 6 类概率柱状图；
2. 上传一个 `mp3` 验证 mp3 解析链路（依赖 `imageio-ffmpeg`，应零报错）；
3. 若构建失败，查看 Space 页面的 **Logs** 标签页：
   - 依赖装不上 → 检查 `requirements.txt` 版本是否兼容 HF 的 Python 版本；
   - 端口暴露失败 → 确认 `demo.launch()` 已绑定 `0.0.0.0`（本项目已完成）。

---

## 附录：本地运行与停止

```bash
.venv\Scripts\activate        # Windows 激活虚拟环境
python app.py                 # 启动界面
# 浏览器打开 http://127.0.0.1:7860
```

按 `Ctrl + C` 停止服务（遵循「想用时手动启动、不用即停」的约定，不常驻后台）。
