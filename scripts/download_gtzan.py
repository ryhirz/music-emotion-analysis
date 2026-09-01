# -*- coding: utf-8 -*-
"""
GTZAN 数据集自动下载与解压脚本
==================================================

【这个脚本做什么】
    GTZAN 是音乐曲风分类的经典公开数据集，包含 10 个曲风、每个曲风 100 首、
    每首 30 秒，共 1000 个音频文件。本脚本帮你一键完成「下载 → 解压 → 归位」，
    最终得到标准目录结构：

        项目根目录/
        └── data/
            └── genres_original/
                ├── blues/       （100 个音频）
                ├── classical/   （100 个音频）
                ├── country/
                ├── disco/
                ├── hiphop/
                ├── jazz/
                ├── metal/
                ├── pop/
                ├── reggae/
                └── rock/

【怎么用】
    1) 常规使用（自动选源，国内优先走 hf-mirror 镜像）:
           python scripts/download_gtzan.py

    2) 需要走代理（公司/校园网/科学上网）:
           python scripts/download_gtzan.py --proxy http://127.0.0.1:7890

    3) 你已经手动下好了压缩包，只想让脚本帮你解压归位:
           python scripts/download_gtzan.py --archive D:/Downloads/genres.tar.gz

    4) 想把数据放到别的位置:
           python scripts/download_gtzan.py --data-dir ./mydata

    5) 只想看脚本会做什么、不真下载（演练模式）:
           python scripts/download_gtzan.py --dry-run

【设计要点】
    - 多源自动回退：镜像不可达会自动换下一个源，不会卡死
    - 断点续传：下载中断后重新运行，会从上次断开的地方继续，不浪费流量
    - 安全解压：过滤掉恶意的「../」路径穿越文件（tar 包的经典漏洞）
    - 自动归位：不管压缩包里顶层目录叫 genres / genres_original / Data，
                最终都整理成 ./data/genres_original/
"""

import argparse
import os
import shutil
import sys
import tarfile
import time
from typing import List, Optional, Tuple

# ─────────────────────────────────────────────────────────────
# 【Windows 中文乱码修复】
# Windows 的 cmd/PowerShell 默认用 GBK 编码，直接 print 中文会报
# UnicodeEncodeError。这里强制把标准输出改成 UTF-8，一劳永逸。
# 用 try/except 包起来是因为 Linux/Mac 上可能没有 reconfigure 方法。
# ─────────────────────────────────────────────────────────────
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# requests 是下载用的 HTTP 库；没装的话给出明确提示而不是让用户看懵的报错
try:
    import requests
except ImportError:
    sys.exit("缺少依赖：请先执行  pip install requests  （或按 requirements.txt 一键安装）")


# ═════════════════════════════════════════════════════════════
# 常量配置区（想改行为优先改这里）
# ═════════════════════════════════════════════════════════════

# GTZAN 的 10 个标准曲风，用于解压后校验目录是否完整
GENRES: List[str] = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
]

# 每个曲风期望的文件数（GTZAN 标准是 100 首）
EXPECTED_FILES_PER_GENRE = 100

# 合法的音频扩展名（原始 Marsyas 版是 .au，HuggingFace 版是 .wav）
VALID_EXTENSIONS = (".au", ".wav")

# 【macOS 垃圾文件过滤】
# GTZAN 的官方压缩包是当年用 Mac 打包的，里面混进了 1000 个 AppleDouble 资源 fork 文件，
# 名字形如 "._blues.00000.wav"（每个仅 211 字节，不是真音频），另外还有 ".DS_Store"。
# 这些文件同样以 .wav 结尾，不过滤的话会被误当成音频统计，
# 出现"每个曲风 200 个文件、合计 2000 个"这种吓人的假警报（实际是 100 个 / 1000 个）。
JUNK_FILE_PREFIXES = ("._",)
JUNK_FILE_NAMES = (".ds_store", "__macosx")


def is_junk_file(filename: str) -> bool:
    """
    判断一个文件名是不是 macOS 打包留下的垃圾文件（而非真正音频）。

    参数:
        filename: 文件名（不含路径）

    返回:
        True 表示是垃圾文件，应当跳过
    """
    lowered = filename.lower()
    # .DS_Store 是 Finder 配置，__MACOSX 是系统打包时生成的资源目录
    if lowered in JUNK_FILE_NAMES:
        return True
    # "._xxx.wav" 是 AppleDouble 资源 fork，体积通常只有几百字节
    return lowered.startswith(JUNK_FILE_PREFIXES)


# 【下载源候选列表】(来源说明, 完整直链)
# 脚本会从上往下依次尝试，第一个能连通的就用来下载。
# 顺序说明：
#   1. hf-mirror.com 是 HuggingFace 的国内镜像，国内通常最快最稳 → 放第一
#   2. huggingface.co 官方源，国外网络环境用它
#   3. opihi.cs.uvic.ca 是 Marsyas 实验室的原始发布源，年代久远经常挂
SOURCES: List[Tuple[str, str]] = [
    (
        "HuggingFace 国内镜像（hf-mirror，国内推荐）",
        "https://hf-mirror.com/datasets/marsyas/gtzan/resolve/main/data/genres.tar.gz",
    ),
    (
        "HuggingFace 官方源",
        "https://huggingface.co/datasets/marsyas/gtzan/resolve/main/data/genres.tar.gz",
    ),
    (
        "Marsyas 实验室原始源（常失效）",
        "https://opihi.cs.uvic.ca/sound/genres.tar.gz",
    ),
]

# 下载块大小：1MB。太大进度条卡、太小 Python 循环开销大，1MB 是经验甜点值
CHUNK_SIZE = 1024 * 1024

# 单次请求超时（秒）：连接超时 15 秒，读取超时 60 秒
TIMEOUT = (15, 60)

# 【每个下载源的最大重试次数】
# 为什么需要：大文件下载经常在 90%+ 处断流（实测本项目就断在 96%）。
# 断流后文件已经下了 1GB 多，此时应该"原地断点重试"，而不是换源从头再来。
# 这里给每个源 5 次机会，每次都用 Range 续传，只有连续 5 次都失败才换下一个源。
MAX_RETRIES_PER_SOURCE = 5

# 每次重试前等待的秒数（给网络一点恢复时间，也避免把服务器打太狠）
RETRY_WAIT_SECONDS = 3


# ═════════════════════════════════════════════════════════════
# 工具函数
# ═════════════════════════════════════════════════════════════

def format_bytes(num_bytes: float) -> str:
    """把字节数转成人类可读的格式，例如 1226192050 → '1.14 GB'。"""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


def print_progress(downloaded: int, total: int, speed: float, bar_len: int = 30) -> None:
    """
    在终端打印一条动态刷新的进度条（用 \\r 回到行首覆盖上一行）。

    参数:
        downloaded: 已下载字节数
        total:      文件总字节数（未知时为 0）
        speed:      当前下载速度（字节/秒）
    """
    if total > 0:
        percent = downloaded / total
        filled = int(bar_len * percent)
        bar = "█" * filled + "░" * (bar_len - filled)
        # end="" 表示不换行，配合 \r 实现原地刷新
        print(
            f"\r   [{bar}] {percent*100:5.1f}%  "
            f"{format_bytes(downloaded)}/{format_bytes(total)}  "
            f"@ {format_bytes(speed)}/s   ",
            end="",
            flush=True,
        )
    else:
        # 服务器没返回 Content-Length（总大小未知），只显示已下载量
        print(
            f"\r   已下载 {format_bytes(downloaded)}  @ {format_bytes(speed)}/s   ",
            end="",
            flush=True,
        )


def download_file(url: str, dest_path: str, proxies: Optional[dict],
                  description: str = "") -> bool:
    """
    流式下载一个文件，带进度条 + 断点续传。

    参数:
        url:        文件直链
        dest_path:  保存到本地的路径
        proxies:    代理字典，例如 {"http": "http://127.0.0.1:7890",
                                   "https": "http://127.0.0.1:7890"}
        description: 来源说明（仅用于打印）

    返回:
        True 表示下载成功；False 表示失败（网络错误 / 服务器拒绝）
    """
    print(f"\n▶ 尝试下载源：{description}")
    print(f"  URL: {url}")

    # ── 断点续传准备 ──
    # 如果目标文件已存在，说明上次没下完。我们读取已有大小，
    # 通过 HTTP Range 头告诉服务器「从第 N 字节开始给我」。
    resumed_bytes = 0
    headers = {}
    if os.path.exists(dest_path):
        resumed_bytes = os.path.getsize(dest_path)
        if resumed_bytes > 0:
            headers["Range"] = f"bytes={resumed_bytes}-"
            print(f"  发现未完成的下载（{format_bytes(resumed_bytes)}），尝试断点续传…")

    try:
        # stream=True 关键：不把整个 1.2GB 读进内存，而是一块一块地取
        response = requests.get(
            url, stream=True, headers=headers,
            timeout=TIMEOUT, proxies=proxies,
        )
        # 服务器说"我支持续传并接受了 Range" → 206 Partial Content
        # 服务器不支持或文件已完整 → 200，此时要从头下
        if response.status_code == 206:
            # 服务器接受了断点续传。Content-Range 形如 "bytes 1024-2047/2048"，
            # 斜杠后面是文件总大小，已下载量仍然是 resumed_bytes。
            mode = "ab"  # append：追加到已有文件末尾
            content_range = response.headers.get("Content-Range", "")
            total = int(content_range.split("/")[-1]) if "/" in content_range else 0
        elif response.status_code == 200:
            mode = "wb"  # 覆盖重写
            resumed_bytes = 0
            total = int(response.headers.get("Content-Length", 0) or 0)
        else:
            print(f"  ✗ 服务器返回异常状态码：{response.status_code}")
            return False

        print(f"  文件大小：{format_bytes(total) if total else '未知'}")

        downloaded = resumed_bytes
        start_time = time.time()
        last_print_time = start_time

        # 分块写入磁盘。iter_content 每次给 CHUNK_SIZE 字节
        with open(dest_path, mode) as file_handle:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue  # 过滤掉保持连接的空块
                file_handle.write(chunk)
                downloaded += len(chunk)

                # 进度条每 0.2 秒刷新一次就够了，刷新太频繁反而拖慢下载
                now = time.time()
                elapsed = now - start_time
                if now - last_print_time >= 0.2:
                    speed = (downloaded - resumed_bytes) / elapsed if elapsed > 0 else 0
                    print_progress(downloaded, total, speed)
                    last_print_time = now

        # 下载结束，补一次 100% 的进度条然后换行
        elapsed = time.time() - start_time
        speed = (downloaded - resumed_bytes) / elapsed if elapsed > 0 else 0
        print_progress(downloaded, total, speed)
        print()  # 换行，避免后续输出跟进度条挤在同一行

        # 完整性校验：实际大小和服务器声明的大小必须一致
        if total > 0:
            actual = os.path.getsize(dest_path)
            if actual != total:
                print(f"  ✗ 文件不完整：期望 {format_bytes(total)}，实际 {format_bytes(actual)}")
                print("    可能是网络中断。重新运行本脚本可继续断点续传。")
                return False

        print(f"  ✓ 下载完成，用时 {elapsed:.1f} 秒")
        return True

    except requests.exceptions.ProxyError:
        print("  ✗ 代理连接失败：请检查 --proxy 参数是否正确、代理软件是否已启动")
        return False
    except requests.exceptions.SSLError:
        print("  ✗ SSL 证书错误：可尝试加 --proxy 走代理，或检查系统时间是否正确")
        return False
    except requests.exceptions.ConnectionError as err:
        print(f"  ✗ 连接失败（网络不通或被墙）：{type(err).__name__}")
        return False
    except requests.exceptions.Timeout:
        print("  ✗ 连接超时：该下载源可能已失效，脚本会自动重试")
        return False
    except requests.exceptions.ChunkedEncodingError as err:
        # 这是大文件下载最常见的错误：连接在中途被掐断。
        # 已经写入磁盘的部分是有效的，重新运行即可续传。
        print(f"  ✗ 下载中断（连接被服务器掐断）：{str(err)[:120]}")
        print("     已下载的部分不会丢失，将自动断点续传。")
        return False
    except Exception as err:
        print(f"  ✗ 未知错误：{type(err).__name__}: {str(err)[:200]}")
        return False


def safe_extract_tar(tar_path: str, extract_to: str) -> None:
    """
    安全解压 .tar.gz 文件。

    【为什么要"安全"解压】
    tar 包里每一项的相对路径可以被恶意构造成 "../../Windows/xxx"，
    普通 extractall 会把文件写到目标目录之外，覆盖系统文件（CVE-2007-4559
    "tar 目录穿越漏洞"）。下载来的数据集虽然一般无害，但养成好习惯很重要。

    Python 3.12+ 的 tarfile 提供了 filter="data" 参数自动拦截这类条目；
    老版本没有这个参数，我们手动逐个检查路径。
    """
    print(f"\n▶ 正在解压到：{extract_to}")
    os.makedirs(extract_to, exist_ok=True)

    # ── 先过滤掉 macOS 垃圾文件 ──
    # 做法：只保留"是普通文件 且 文件名不是垃圾"的条目，其余直接不解压。
    # 这样最终的数据目录是干净的，后面统计时不会误报。
    with tarfile.open(tar_path, "r:gz") as tar:
        skipped_junk = 0
        for member in tar.getmembers():
            basename = os.path.basename(member.name)
            if member.isfile() and is_junk_file(basename):
                skipped_junk += 1
        if skipped_junk > 0:
            print(f"  检测到 {skipped_junk} 个 macOS 垃圾文件（._* / .DS_Store），解压时将跳过")

    with tarfile.open(tar_path, "r:gz") as tar:
        try:
            # Python 3.12+：用官方的数据过滤器，最省心
            # filter="data" 已经会自动丢掉大部分危险条目，我们在此基础上
            # 再加一层垃圾文件过滤（通过 members 参数显式指定保留哪些）
            safe_members = [
                m for m in tar.getmembers()
                if not (m.isfile() and is_junk_file(os.path.basename(m.name)))
            ]
            tar.extractall(path=extract_to, members=safe_members, filter="data")
        except TypeError:
            # 老版本 Python：手动过滤危险路径 + 垃圾文件
            safe_members = []
            for member in tar.getmembers():
                member_path = os.path.abspath(os.path.join(extract_to, member.name))
                # 判断解压后的绝对路径是否仍在目标目录之内（防目录穿越）
                if not member_path.startswith(os.path.abspath(extract_to) + os.sep):
                    print(f"  ⚠ 跳过可疑文件（路径穿越）：{member.name}")
                    continue
                # 跳过 macOS 垃圾文件
                if member.isfile() and is_junk_file(os.path.basename(member.name)):
                    continue
                safe_members.append(member)
            tar.extractall(path=extract_to, members=safe_members)

    print("  ✓ 解压完成")


def find_genres_root(search_dir: str) -> Optional[str]:
    """
    在解压后的目录树里，找出真正装着 10 个曲风子文件夹的那一层。

    为什么要找？因为不同来源的压缩包顶层结构不一样：
        Marsyas 原始包   → genres/blues, genres/rock, ...
        HuggingFace 包   → data/genres/blues, ...
        Kaggle 版本      → Data/genres_original/blues, ...
    我们不写死路径，而是"搜索"，这样换源也不会崩。

    返回: 找到的目录绝对路径；找不到返回 None
    """
    # os.walk 会递归遍历所有子目录；这里只要找到就立刻返回
    for current_dir, subdirs, _files in os.walk(search_dir):
        # 把当前目录下的子目录名转成小写集合，方便不区分大小写比对
        subdir_names_lower = {d.lower() for d in subdirs}
        # 统计 10 个标准曲风里，有几个出现在当前目录下
        matched = sum(1 for genre in GENRES if genre.lower() in subdir_names_lower)
        # 命中 ≥8 个就认定是我们要找的目录
        # （用 8 而不是 10，是容忍个别压缩包曲风命名略有出入）
        if matched >= 8:
            return current_dir
    return None


def normalize_to_genres_original(data_dir: str) -> Optional[str]:
    """
    把找到的曲风根目录，统一搬移/重命名为  data/genres_original/ 。

    返回: 最终的 genres_original 绝对路径；失败返回 None
    """
    # 先解压到一个临时目录，避免污染
    temp_extract_dir = os.path.join(data_dir, "_tmp_extract")
    found_root = find_genres_root(temp_extract_dir)

    if found_root is None:
        print("\n✗ 在解压结果中找不到包含 10 个曲风子目录的文件夹。")
        print("  压缩包结构可能异常，请检查下载的文件是否完整。")
        return None

    final_path = os.path.join(data_dir, "genres_original")

    # 如果目标位置已经存在，先删掉旧的，保证是干净的一份
    if os.path.exists(final_path):
        print(f"\n  发现已存在的目录 {final_path}，先删除以便重新生成…")
        shutil.rmtree(final_path)

    # 把找到的目录整体移动/重命名到最终位置
    shutil.move(found_root, final_path)

    # 清理临时解压目录（里面只剩下空壳文件夹了）
    if os.path.exists(temp_extract_dir):
        shutil.rmtree(temp_extract_dir, ignore_errors=True)

    return final_path


def verify_structure(genres_original_path: str) -> bool:
    """
    解压完成后做一次快速体检：10 个曲风目录在不在、每个目录大概多少文件。

    返回: True 表示结构基本正常
    """
    print("\n" + "=" * 60)
    print("数据集结构体检")
    print("=" * 60)

    all_ok = True
    total_files = 0

    for genre in GENRES:
        genre_dir = os.path.join(genres_original_path, genre)
        if not os.path.isdir(genre_dir):
            print(f"  ✗ {genre:<12} 目录缺失！")
            all_ok = False
            continue

        # 只统计合法音频扩展名（.au / .wav），忽略系统自动生成的隐藏文件
        audio_files = [
            f for f in os.listdir(genre_dir)
            if f.lower().endswith(VALID_EXTENSIONS) and not is_junk_file(f)
        ]
        total_files += len(audio_files)

        # 文件数不等于 100 就打个警告（不阻塞，因为有些镜像确实少几首）
        if len(audio_files) == EXPECTED_FILES_PER_GENRE:
            status = "✓"
        else:
            status = "⚠"
            all_ok = False
        print(f"  {status} {genre:<12} {len(audio_files):>4} 个音频文件")

    print("-" * 60)
    print(f"  合计：{total_files} 个音频文件（标准应为 1000 个）")

    if all_ok:
        print("  ✓ 结构完整，数据集已就绪！")
    else:
        print("  ⚠ 有项目不达标（详见上方标记）。可运行 verify_dataset.py 做详细检查。")

    return all_ok


def print_manual_guide(data_dir: str) -> None:
    """
    所有自动下载源都失败时，打印人工补救方案。
    """
    print("\n" + "=" * 60)
    print("所有自动下载源都失败了 —— 备选方案")
    print("=" * 60)
    print("""
方案 A：用 Kaggle 官方数据集（推荐，最稳定）
    1. 注册/登录 https://www.kaggle.com
    2. 打开 https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
    3. 点右上角 Download 下载 zip（约 1.2 GB）
    4. 把下载到的压缩包放到项目目录下，然后执行：
           python scripts/download_gtzan.py --archive <你下载的压缩包路径>

方案 B：用 HuggingFace 网页手动下载
    1. 浏览器打开（国内推荐用镜像域名）：
           https://hf-mirror.com/datasets/marsyas/gtzan/tree/main/data
    2. 点击 genres.tar.gz 下载
    3. 放到项目目录后执行：
           python scripts/download_gtzan.py --archive genres.tar.gz

方案 C：走代理 / 换个网络
    很多高校和公司网络会拦截境外大文件下载。可先确认代理端口后重试：
           python scripts/download_gtzan.py --proxy http://127.0.0.1:7890
    （端口请替换成你代理软件实际监听的端口）

方案 D：直接用 HuggingFace datasets 库（备选代码路径）
        pip install datasets
        python -c "from datasets import load_dataset; ds = load_dataset('marsyas/gtzan', split='train')"
    这条命令会把数据缓存到 ~/.cache/huggingface/，之后可用 ds[i]['audio'] 读取。
    注意：这条路得到的是 HF 的 parquet 格式，不是 genres_original 目录结构，
          后续特征提取代码需要相应调整，因此不作为首选。
""")
    print(f"（目标目录：{os.path.abspath(data_dir)}）")


def try_kaggle_download(data_dir: str) -> Optional[str]:
    """
    尝试调用 Kaggle 官方命令行工具下载数据集。

    前置条件：用户已配置 ~/.kaggle/kaggle.json（在 Kaggle 账号设置里生成）
    返回: 下载到的压缩包路径；失败返回 None
    """
    print("\n▶ 尝试 Kaggle 官方 API…")
    try:
        import subprocess
        result = subprocess.run(
            ["kaggle", "datasets", "download",
             "-d", "andradaolteanu/gtzan-dataset-music-genre-classification",
             "-p", data_dir],
            capture_output=True, text=True, timeout=1800,
        )
        if result.returncode != 0:
            print("  ✗ Kaggle 下载失败（多半是没配置 API Token）")
            print(f"    错误输出：{result.stderr.strip()[:300]}")
            return None

        # Kaggle 下载的一般是 .zip，找到它
        for fname in os.listdir(data_dir):
            if fname.endswith(".zip"):
                print(f"  ✓ Kaggle 下载成功：{fname}")
                return os.path.join(data_dir, fname)
        return None
    except FileNotFoundError:
        print("  ✗ 未安装 kaggle 命令行工具（pip install kaggle）")
        return None
    except Exception as err:
        print(f"  ✗ Kaggle 调用出错：{err}")
        return None


# ═════════════════════════════════════════════════════════════
# 主流程
# ═════════════════════════════════════════════════════════════

def main() -> int:
    # ── 解析命令行参数 ──
    parser = argparse.ArgumentParser(
        description="GTZAN 音乐数据集自动下载与解压工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", default="./data",
        help="数据存放根目录，默认 ./data（最终结构为 ./data/genres_original/）",
    )
    parser.add_argument(
        "--archive", default=None,
        help="如果你已手动下载好压缩包，用这个参数指定路径，脚本将跳过下载直接解压",
    )
    parser.add_argument(
        "--proxy", default=None,
        help="代理地址，例如 http://127.0.0.1:7890",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="演练模式：只打印将要执行的操作，不真的下载",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("GTZAN 数据集下载工具")
    print("=" * 60)

    data_dir = os.path.abspath(args.data_dir)
    final_path = os.path.join(data_dir, "genres_original")

    # ── 演练模式：只打印计划 ──
    if args.dry_run:
        print("\n[演练模式] 将要执行的操作：")
        print(f"  1. 下载源候选（按顺序尝试）：")
        for desc, url in SOURCES:
            print(f"       - {desc}: {url}")
        print(f"  2. 下载到：{os.path.join(data_dir, 'genres.tar.gz')}")
        print(f"  3. 解压并整理到：{final_path}")
        if args.proxy:
            print(f"  4. 使用代理：{args.proxy}")
        print("\n（演练模式未做任何实际操作）")
        return 0

    # ── 已经解压好了？直接跳过，避免重复劳动 ──
    if os.path.isdir(final_path):
        existing = sum(
            len([f for f in os.listdir(os.path.join(final_path, g))
                 if f.lower().endswith(VALID_EXTENSIONS) and not is_junk_file(f)])
            for g in GENRES if os.path.isdir(os.path.join(final_path, g))
        )
        if existing > 0:
            print(f"\n检测到数据集已存在：{final_path}")
            print(f"   共 {existing} 个音频文件")
            print("   如需重新下载，请先手动删除该目录。")
            verify_structure(final_path)
            return 0

    os.makedirs(data_dir, exist_ok=True)

    # ── 组装代理配置 ──
    proxies = None
    if args.proxy:
        proxies = {"http": args.proxy, "https": args.proxy}
        print(f"\n已启用代理：{args.proxy}")

    # ═══ 第一步：拿到压缩包 ═══
    archive_path = args.archive

    if archive_path:
        # 用户已手动下载，跳过网络环节
        archive_path = os.path.abspath(archive_path)
        if not os.path.exists(archive_path):
            print(f"\n✗ 指定的压缩包不存在：{archive_path}")
            return 1
        print(f"\n✓ 使用本地已有压缩包：{archive_path}（{format_bytes(os.path.getsize(archive_path))}）")
    else:
        # 依次尝试各个下载源；每个源断流后会原地断点重试，而不是立刻换源
        tar_path = os.path.join(data_dir, "genres.tar.gz")
        downloaded_ok = False

        for index, (description, url) in enumerate(SOURCES, start=1):
            for attempt in range(1, MAX_RETRIES_PER_SOURCE + 1):
                retry_hint = f"（第 {attempt}/{MAX_RETRIES_PER_SOURCE} 次尝试）"
                print(f"\n[{index}/{len(SOURCES)}] {retry_hint} " + "-" * 30)

                if download_file(url, tar_path, proxies, description):
                    downloaded_ok = True
                    archive_path = tar_path
                    break

                if attempt < MAX_RETRIES_PER_SOURCE:
                    print(f"  等待 {RETRY_WAIT_SECONDS} 秒后断点重试…")
                    time.sleep(RETRY_WAIT_SECONDS)
                else:
                    print("  该源重试次数已用尽，切换下一个下载源…")

            if downloaded_ok:
                break

        if not downloaded_ok:
            # 自动源全挂 → 尝试 Kaggle → 仍失败则打印人工指南
            print("\n" + "!" * 60)
            print("全部自动下载源均失败")
            print("!" * 60)
            kaggle_result = try_kaggle_download(data_dir)
            if kaggle_result:
                archive_path = kaggle_result
            else:
                print_manual_guide(data_dir)
                return 1

    # ═══ 第二步：解压并整理目录 ═══
    print("\n" + "=" * 60)
    print("解压与目录整理")
    print("=" * 60)

    try:
        # 统一解压到临时目录，再由 normalize 归位，避免顶层目录名不一致的坑
        temp_extract_dir = os.path.join(data_dir, "_tmp_extract")
        if os.path.exists(temp_extract_dir):
            shutil.rmtree(temp_extract_dir, ignore_errors=True)
        safe_extract_tar(archive_path, temp_extract_dir)

        result_path = normalize_to_genres_original(data_dir)
        if result_path is None:
            return 1

    except tarfile.ReadError:
        print("\n✗ 压缩包损坏或不是有效的 tar.gz 格式。")
        print("  建议：删除后重新运行本脚本下载。")
        return 1
    except Exception as err:
        print(f"\n✗ 解压过程出错：{type(err).__name__}: {err}")
        return 1

    # ═══ 第三步：结构体检 + 收尾提示 ═══
    verify_structure(result_path)

    print("\n" + "=" * 60)
    print("全部完成！")
    print("=" * 60)
    print(f"   数据集位置：{result_path}")
    print(f"   压缩包备份：{archive_path}（可删除以节省空间）")
    print("\n下一步：运行数据验证脚本，检查文件是否都能被 librosa 正常读取")
    print("       python scripts/verify_dataset.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
