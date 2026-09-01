# -*- coding: utf-8 -*-
"""
GTZAN 数据集完整性验证脚本
==================================================

【这个脚本做什么】
    下载完 GTZAN 后，别急着跑模型 —— 先确认数据是好的。这个脚本会做全面体检：

    1. 目录结构检查：10 个曲风文件夹是否齐全
    2. 文件数量检查：每个曲风是不是标准的 100 个文件
    3. 格式检查：文件后缀是不是 .au 或 .wav（GTZAN 两种版本都有）
    4. 损坏检查：用 librosa 逐个真正加载每个音频，加载失败的就是坏文件
    5. 空文件检查：找出 0 字节的"假文件"
    6. 统计概览：总文件数、各曲风文件数、时长的最短/最长/平均值、采样率分布

【怎么用】
    完整检查全部 1000 个文件（约 3-10 分钟，取决于 CPU）:
        python scripts/verify_dataset.py

    先快速试跑 20 个文件，确认脚本能跑通:
        python scripts/verify_dataset.py --limit 20

    数据放在别的位置:
        python scripts/verify_dataset.py --data-dir ./mydata

    把详细结果导出成 CSV 方便后续分析:
        python scripts/verify_dataset.py --report data/dataset_report.csv

    确认无误后删除损坏文件（危险操作，需显式加参数）:
        python scripts/verify_dataset.py --delete-broken

    删掉 GTZAN 压缩包自带的 macOS 垃圾文件（._* / .DS_Store，共约 1000 个）:
        python scripts/verify_dataset.py --clean-junk

【为什么"损坏检查"很重要】
    GTZAN 是 2001 年的老数据集，网络上流传的版本里普遍混有少量损坏文件
    （最出名的是 jazz 曲风里有个文件是坏的）。如果不在这一步揪出来，
    后面跑特征提取时会突然崩溃，而且报错信息往往指向不相关的代码，
    排查起来非常浪费时间。
"""

import argparse
import os
import shutil
import sys
import time
from typing import Dict, List, Optional

# ── Windows 中文输出修复（cmd 默认 GBK，直接 print 中文会崩）──
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# ── 依赖检查：给出友好提示，而不是让用户面对一堆红色 traceback ──
try:
    import librosa
except ImportError:
    sys.exit(
        "缺少 librosa。请先安装依赖：\n"
        "    pip install -r requirements.txt\n"
        "或单独安装：\n"
        "    pip install librosa==0.10.2.post1"
    )

try:
    import numpy as np
    import pandas as pd
except ImportError as err:
    sys.exit(f"缺少依赖（{err}）。请执行：pip install -r requirements.txt")

# tqdm 用来显示进度条，没装也不影响功能，降级为普通打印
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ═════════════════════════════════════════════════════════════
# 常量配置
# ═════════════════════════════════════════════════════════════

# GTZAN 标准的 10 个曲风
GENRES: List[str] = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
]

# 每个曲风应有的文件数
EXPECTED_FILES_PER_GENRE = 100

# 合法音频扩展名（原始版 .au，HuggingFace 版 .wav）
VALID_EXTENSIONS = (".au", ".wav")

# 【macOS 垃圾文件过滤】
# GTZAN 官方压缩包是当年用 Mac 打包的，里面混了 1000 个 AppleDouble 资源 fork 文件，
# 名字形如 "._blues.00000.wav"（每个仅 211 字节，不是真音频），还有 ".DS_Store"。
# 它们同样以 .wav 结尾，不过滤会被误统计成"每个曲风 200 个文件"的假警报。
JUNK_FILE_PREFIXES = ("._",)
JUNK_FILE_NAMES = (".ds_store", "__macosx")


def is_junk_file(filename: str) -> bool:
    """
    判断文件名是否为 macOS 打包遗留的垃圾文件（而非真正音频）。

    参数:
        filename: 文件名（不含路径）

    返回:
        True 表示是垃圾文件，应跳过
    """
    lowered = filename.lower()
    if lowered in JUNK_FILE_NAMES:
        return True
    return lowered.startswith(JUNK_FILE_PREFIXES)

# 统一重采样到这个采样率加载。GTZAN 原生就是 22050Hz，所以这里不会产生
# 额外的重采样开销，同时保证所有音频特征维度一致。
TARGET_SAMPLE_RATE = 22050

# GTZAN 每个文件标准时长是 30 秒，允许 ±0.5 秒的误差
EXPECTED_DURATION = 30.0
DURATION_TOLERANCE = 0.5


# ═════════════════════════════════════════════════════════════
# 核心检查函数
# ═════════════════════════════════════════════════════════════

def scan_dataset(genres_dir: str, limit: Optional[int] = None) -> List[Dict]:
    """
    遍历数据集目录，逐个用 librosa 加载音频，收集每一条记录。

    参数:
        genres_dir: genres_original 目录路径
        limit:      只检查前 N 个文件（调试用）；None 表示全查

    返回:
        记录列表，每条是一个 dict，包含文件名/曲风/时长/采样率/状态等信息
    """
    records: List[Dict] = []
    checked_count = 0
    junk_count = 0

    print("\n开始逐个加载音频文件（这一步是真正的完整性检查）…")
    print("提示：1000 个文件通常需要 3-10 分钟，请耐心等待。\n")

    # 用 tqdm 包一层进度条；没装 tqdm 就用原始迭代器
    genre_iterator = tqdm(GENRES, desc="曲风进度", unit="genre") if HAS_TQDM else GENRES

    for genre in genre_iterator:
        genre_dir = os.path.join(genres_dir, genre)

        # 曲风目录不存在：记录一条"目录缺失"，继续检查其他曲风
        if not os.path.isdir(genre_dir):
            records.append({
                "genre": genre, "filename": "(目录缺失)", "extension": "",
                "status": "目录缺失", "duration_sec": None,
                "sample_rate": None, "size_bytes": None, "error": "文件夹不存在",
            })
            continue

        # 列出该目录下所有文件并排序，保证每次运行结果顺序一致
        all_files = sorted(os.listdir(genre_dir))

        for filename in all_files:
            file_path = os.path.join(genre_dir, filename)

            # 跳过子目录（解压时可能生成 __MACOSX 之类的目录）
            if not os.path.isfile(file_path):
                continue

            # 跳过 macOS 垃圾文件（._xxx.wav / .DS_Store）
            # 这些不是真音频，只计数不进入检查流程
            if is_junk_file(filename):
                junk_count += 1
                continue

            # 如果设置了 limit 且已达上限，停止收集
            if limit is not None and checked_count >= limit:
                break

            checked_count += 1

            # ── 基础信息：扩展名、文件大小 ──
            extension = os.path.splitext(filename)[1].lower()
            try:
                size_bytes = os.path.getsize(file_path)
            except OSError:
                size_bytes = None

            # ── 检查 1：扩展名是否合法 ──
            if extension not in VALID_EXTENSIONS:
                records.append({
                    "genre": genre, "filename": filename, "extension": extension,
                    "status": "格式异常", "duration_sec": None,
                    "sample_rate": None, "size_bytes": size_bytes,
                    "error": f"扩展名不是 .au/.wav（实际：{extension or '无扩展名'}）",
                })
                continue

            # ── 检查 2：是否是 0 字节空文件 ──
            if size_bytes == 0:
                records.append({
                    "genre": genre, "filename": filename, "extension": extension,
                    "status": "空文件", "duration_sec": None,
                    "sample_rate": None, "size_bytes": 0,
                    "error": "文件大小为 0 字节",
                })
                continue

            # ── 检查 3：真正用 librosa 加载（核心的损坏检测）──
            try:
                # sr=TARGET_SAMPLE_RATE 统一采样率；mono=True 统一为单声道
                # 这两项统一后，后面提取的特征维度才会一致
                audio_data, actual_sr = librosa.load(
                    file_path, sr=TARGET_SAMPLE_RATE, mono=True
                )
                # 时长 = 采样点总数 / 每秒采样点数
                duration = len(audio_data) / float(actual_sr)

                # 顺便看看原始采样率（librosa.get_samplerate 只读文件头，很快）
                try:
                    native_sr = librosa.get_samplerate(file_path)
                except Exception:
                    native_sr = None

                # 判断时长是否符合 30 秒的标准
                if abs(duration - EXPECTED_DURATION) > DURATION_TOLERANCE:
                    status = "时长异常"
                    error = (f"时长 {duration:.2f}s，"
                             f"偏离标准 {EXPECTED_DURATION}s 超过 {DURATION_TOLERANCE}s")
                else:
                    status = "正常"
                    error = ""

                records.append({
                    "genre": genre, "filename": filename, "extension": extension,
                    "status": status, "duration_sec": round(duration, 3),
                    "sample_rate": native_sr, "size_bytes": size_bytes,
                    "error": error,
                })

            except Exception as err:
                # 任何加载失败都归为"损坏文件"，把错误原因记下来方便排查
                records.append({
                    "genre": genre, "filename": filename, "extension": extension,
                    "status": "损坏", "duration_sec": None,
                    "sample_rate": None, "size_bytes": size_bytes,
                    "error": f"{type(err).__name__}: {str(err)[:200]}",
                })

        # limit 达到上限后，外层循环也要跳出
        if limit is not None and checked_count >= limit:
            break

    if junk_count > 0:
        print(f"\n（已自动忽略 {junk_count} 个 macOS 垃圾文件：._* / .DS_Store，"
              f"它们不是真音频。想彻底删掉可加 --clean-junk 参数）")

    return records


def print_overview(records: List[Dict], genres_dir: str) -> Dict[str, int]:
    """
    打印数据集概览：总文件数、各曲风文件数、格式分布、时长统计、损坏清单。

    参数:
        records:     scan_dataset 收集到的记录列表
        genres_dir:  数据集路径（用于打印表头）

    返回:
        统计摘要字典（各状态的数量）
    """
    dataframe = pd.DataFrame(records)

    # ═══ 板块 1：总体概览 ═══
    print("\n" + "=" * 66)
    print("一、总体概览")
    print("=" * 66)
    print(f"   数据集路径：{os.path.abspath(genres_dir)}")
    print(f"   扫描文件总数：{len(dataframe)}")

    # 把"目录缺失"这种伪记录排除掉，只统计真实文件
    real_files = dataframe[dataframe["filename"] != "(目录缺失)"]
    print(f"   实际音频文件数：{len(real_files)}")

    # 统计各种状态的数量
    status_counts = dataframe["status"].value_counts().to_dict()
    normal_count = status_counts.get("正常", 0)
    print(f"   状态正常：{normal_count} 个")

    # ═══ 板块 2：各曲风文件数 ═══
    print("\n" + "=" * 66)
    print("二、各曲风文件数量（标准应为 100 个/曲风）")
    print("=" * 66)

    genre_rows = []
    for genre in GENRES:
        genre_data = real_files[real_files["genre"] == genre]
        total = len(genre_data)
        valid = len(genre_data[genre_data["status"] == "正常"])
        broken = len(genre_data[genre_data["status"] == "损坏"])
        abnormal = total - valid - broken

        # 用符号直观标出是否达标
        mark = "✓" if (total == EXPECTED_FILES_PER_GENRE and broken == 0) else "⚠"
        genre_rows.append({
            "状态": mark, "曲风": genre, "文件数": total,
            "正常": valid, "损坏": broken, "其他异常": abnormal,
        })

    genre_summary = pd.DataFrame(genre_rows)
    # to_string(index=False) 打印成整齐的表格，不带 pandas 的索引列
    print(genre_summary.to_string(index=False))
    print(f"\n   合计：{len(real_files)} 个文件"
          f"（标准 1000 个 = 10 曲风 × 100 首）")

    # ═══ 板块 3：文件格式分布 ═══
    print("\n" + "=" * 66)
    print("三、文件格式分布")
    print("=" * 66)
    extension_counts = real_files["extension"].value_counts()
    for ext, count in extension_counts.items():
        is_valid = "（合法）" if ext in VALID_EXTENSIONS else "（异常！应为 .au 或 .wav）"
        print(f"   {ext or '(无扩展名)':<8} {count:>5} 个  {is_valid}")

    # 采样率分布（这里看的是文件原始采样率）
    print("\n   原始采样率分布：")
    sr_counts = real_files["sample_rate"].value_counts(dropna=True)
    for sr, count in sr_counts.items():
        print(f"     {int(sr):>7} Hz  {count:>5} 个")

    # ═══ 板块 4：时长统计 ═══
    # 只对状态正常、能算出时长的文件做统计
    valid_data = real_files[real_files["duration_sec"].notna()]

    print("\n" + "=" * 66)
    print("四、音频时长统计（GTZAN 标准为 30.00 秒）")
    print("=" * 66)

    if len(valid_data) > 0:
        durations = valid_data["duration_sec"].astype(float)
        print(f"   最短：{durations.min():.2f} 秒")
        print(f"   最长：{durations.max():.2f} 秒")
        print(f"   平均：{durations.mean():.2f} 秒")
        print(f"   中位数：{durations.median():.2f} 秒")
        print(f"   标准差：{durations.std():.2f} 秒")

        print("\n   按曲风细分：")
        # groupby + agg 一次性算出每个曲风的多个统计量
        per_genre = valid_data.groupby("genre")["duration_sec"].agg(
            文件数="count", 最短="min", 最长="max", 平均="mean"
        ).round(2)
        print(per_genre.to_string())
    else:
        print("   ⚠ 没有可统计的有效音频，请检查数据集是否解压正确")

    # ═══ 板块 5：问题文件清单 ═══
    problem_files = dataframe[dataframe["status"] != "正常"]

    print("\n" + "=" * 66)
    print("五、问题文件清单")
    print("=" * 66)

    if len(problem_files) == 0:
        print("   ✓ 未发现任何问题文件，数据集完全健康！")
    else:
        print(f"   发现 {len(problem_files)} 个问题文件：\n")
        for _, row in problem_files.iterrows():
            print(f"   [{row['status']}] {row['genre']}/{row['filename']}")
            if row["error"]:
                # 错误信息可能很长，截断到 150 字符避免刷屏
                print(f"            原因：{str(row['error'])[:150]}")

        # 针对 GTZAN 的经典问题给具体建议
        broken_count = len(dataframe[dataframe["status"] == "损坏"])
        if broken_count > 0:
            print(f"\n   💡 建议：这 {broken_count} 个损坏文件会在特征提取时导致程序崩溃。")
            print("      推荐做法是在特征提取脚本里 try/except 跳过它们（不影响模型训练），")
            print("      或者确认后执行：python scripts/verify_dataset.py --delete-broken")

    return status_counts


def clean_junk_files(genres_dir: str) -> int:
    """
    删除数据集里的 macOS 垃圾文件（._* / .DS_Store / __MACOSX）。

    这些文件是 GTZAN 官方压缩包当年用 Mac 打包时混进去的，每个仅约 211 字节，
    不是真音频，留着只会干扰统计和特征提取。删除它们不会影响任何正常音频。

    返回: 删除的文件数量
    """
    removed = 0
    for current_dir, subdirs, files in os.walk(genres_dir):
        # 顺手把 __MACOSX 资源目录也标记为待删
        for dirname in list(subdirs):
            if dirname.lower() == "__macosx":
                shutil.rmtree(os.path.join(current_dir, dirname), ignore_errors=True)
                print(f"  已删除目录：{os.path.join(current_dir, dirname)}")
        for filename in files:
            if is_junk_file(filename):
                try:
                    os.remove(os.path.join(current_dir, filename))
                    removed += 1
                except OSError as err:
                    print(f"  ✗ 删除失败 {filename}：{err}")
    return removed


def delete_broken_files(records: List[Dict], genres_dir: str) -> int:
    """
    删除检测出的损坏文件。

    【安全设计】
    这是不可逆操作，所以：
      - 必须显式传 --delete-broken 才会执行
      - 只删除"损坏"和"空文件"两类（不删"时长异常"，因为那可能仍可用）
      - 删除前会打印完整清单
    """
    broken = [r for r in records if r["status"] in ("损坏", "空文件")]

    if not broken:
        print("\n没有需要删除的损坏文件。")
        return 0

    print("\n" + "!" * 66)
    print(f"即将删除以下 {len(broken)} 个文件（此操作不可恢复）：")
    print("!" * 66)
    for record in broken:
        print(f"   {record['genre']}/{record['filename']}")

    deleted = 0
    for record in broken:
        file_path = os.path.join(genres_dir, record["genre"], record["filename"])
        try:
            os.remove(file_path)
            deleted += 1
        except OSError as err:
            print(f"   ✗ 删除失败 {record['filename']}：{err}")

    print(f"\n已删除 {deleted} 个损坏文件。")
    return deleted


def main() -> int:
    # ── 命令行参数 ──
    parser = argparse.ArgumentParser(
        description="GTZAN 数据集完整性验证工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", default="./data",
        help="数据根目录，默认 ./data（脚本会检查 ./data/genres_original/）",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="只检查前 N 个文件（快速试跑用），例如 --limit 20",
    )
    parser.add_argument(
        "--report", default=None,
        help="把逐文件的详细结果导出为 CSV，例如 --report data/report.csv",
    )
    parser.add_argument(
        "--delete-broken", action="store_true",
        help="删除检测出的损坏/空文件（危险，不可恢复，需显式指定）",
    )
    parser.add_argument(
        "--clean-junk", action="store_true",
        help="删除 macOS 垃圾文件（._* / .DS_Store / __MACOSX），不影响正常音频",
    )
    args = parser.parse_args()

    print("=" * 66)
    print("GTZAN 数据集完整性验证")
    print("=" * 66)

    # ── 定位并检查 genres_original 目录 ──
    genres_dir = os.path.join(os.path.abspath(args.data_dir), "genres_original")

    if not os.path.isdir(genres_dir):
        print(f"\n✗ 找不到数据集目录：{genres_dir}")
        print("\n请先运行下载脚本：")
        print("    python scripts/download_gtzan.py")
        return 1

    print(f"\n数据集目录：{genres_dir}")
    if args.limit:
        print(f"（演练模式：只检查前 {args.limit} 个文件）")

    # ── 可选的垃圾文件清理（放在扫描前，避免干扰统计）──
    if args.clean_junk:
        print("\n▶ 清理 macOS 垃圾文件…")
        removed = clean_junk_files(genres_dir)
        if removed:
            print(f"  ✓ 已删除 {removed} 个垃圾文件")
        else:
            print("  没有发现垃圾文件，数据集很干净")

    # ── 执行扫描 ──
    start_time = time.time()
    records = scan_dataset(genres_dir, limit=args.limit)
    elapsed = time.time() - start_time

    if not records:
        print("\n✗ 没有扫描到任何文件，请检查目录是否为空。")
        return 1

    print(f"\n扫描完成，用时 {elapsed:.1f} 秒")

    # ── 打印概览 ──
    status_counts = print_overview(records, genres_dir)

    # ── 可选的 CSV 导出 ──
    if args.report:
        report_path = os.path.abspath(args.report)
        os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
        pd.DataFrame(records).to_csv(report_path, index=False, encoding="utf-8-sig")
        print(f"\n✓ 详细报告已导出：{report_path}")
        print("  （用 utf-8-sig 编码，Excel 直接打开不会中文乱码）")

    # ── 可选的删除损坏文件 ──
    if args.delete_broken:
        delete_broken_files(records, genres_dir)

    # ── 最终结论 ──
    print("\n" + "=" * 66)
    broken_total = sum(
        v for k, v in status_counts.items() if k in ("损坏", "空文件", "目录缺失")
    )
    if broken_total == 0:
        print("✓ 数据集验证通过，可以开始下一步：音频特征提取")
        return 0
    else:
        print(f"⚠ 数据集存在 {broken_total} 个严重问题（损坏/空文件/目录缺失）")
        print("  建议在特征提取脚本中加入异常跳过逻辑，或先清理这些问题文件。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
