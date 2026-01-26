import os
import sys
import subprocess
from pathlib import Path
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor

# ---------------------
# 配置
# ---------------------
IMG_DIR = Path('/data/cj/MME/crop_with_relative_color_8_17/images')
OUT_DIR = Path('/data/cj/MME/crop_with_relative_color_8_17/images_sky_out')
OUT_DIR.mkdir(parents=True, exist_ok=True)

PYTHON_BIN = sys.executable  # 用当前 python
SCRIPT    = "/data/cj/SkySense-O/demo/demo_8_6.py"
CFG       = "/data/cj/SkySense-O/configs/skysense_o_8_6_demo.yaml"

# 是否跳过已有输出（强烈建议开，避免重复计算）
SKIP_IF_EXISTS = True

# 允许的输入扩展名
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ---------------------
# GPU 探测
# ---------------------
def detect_gpus() -> List[str]:
    """返回可用 GPU 的 ID 列表（字符串）"""
    # 1) 优先尊重外部已设置的 CUDA_VISIBLE_DEVICES（如果你在外层 tmux/shell 里限定了）
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if vis:
        ids = [x for x in vis.split(",") if x != ""]
        if ids:
            return ids

    # 2) 否则通过 torch 探测
    try:
        import torch
        n = torch.cuda.device_count()
        if n and n > 0:
            return [str(i) for i in range(n)]
    except Exception:
        pass

    # 3) 兜底：无 GPU，当成 CPU（仍然返回一个“逻辑 GPU 0”，让流程走通）
    return ["0"]

# ---------------------
# 任务分片
# ---------------------
def list_images(img_dir: Path) -> List[Path]:
    items = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTS]
    items.sort()
    return items

def make_output_path(inp: Path) -> Path:
    # 输出文件沿用原文件名（可按需换后缀/改名）
    return OUT_DIR / inp.name

def shard(items: List[Path], ngpus: int) -> List[List[Path]]:
    """把任务按 stride 方式切成 ngpus 份，确保每块卡有自己的任务队列"""
    buckets = [[] for _ in range(ngpus)]
    for i, x in enumerate(items):
        buckets[i % ngpus].append(x)
    return buckets

# ---------------------
# 核心执行
# ---------------------
def run_one_on_gpu(gpu_id: str, todo: List[Path]):
    """绑定到单个 GPU，顺序执行分配的任务"""
    if not todo:
        print(f"[GPU {gpu_id}] 没有分配到任务")
        return []

    env = os.environ.copy()
    # 关键：让子进程只“看到”这一块卡
    env["CUDA_VISIBLE_DEVICES"] = gpu_id

    failed = []
    for img_path in todo:
        out_path = make_output_path(img_path)

        if SKIP_IF_EXISTS and out_path.exists():
            print(f"[GPU {gpu_id}] SKIP 已存在: {out_path.name}")
            continue

        # 目标目录要保证存在（demo 里通常会创建父目录，这里以防万一）
        out_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            PYTHON_BIN, SCRIPT,
            "--eval-only",
            "--config-file", CFG,
            "--input", str(img_path),
            "--output", str(out_path),
            "--num-gpus", "1"   # 子进程内只用它自己那张卡
        ]
        print(f"[GPU {gpu_id}] RUN  {img_path.name}")
        res = subprocess.run(cmd, env=env)
        if res.returncode != 0:
            print(f"[GPU {gpu_id}] ERR  {img_path.name}")
            failed.append(str(img_path))
        else:
            print(f"[GPU {gpu_id}] DONE {img_path.name} -> {out_path.name}")

    return failed

# ---------------------
# 主入口
# ---------------------
def main():
    imgs = list_images(IMG_DIR)
    if not imgs:
        print(f"没有在 {IMG_DIR} 里找到可处理的图片（支持: {sorted(EXTS)}）")
        return

    gpu_ids = detect_gpus()
    ng = len(gpu_ids)
    print(f"发现可用 GPU: {gpu_ids}（共 {ng} 块）")
    buckets = shard(imgs, ng)

    # 每块卡一个 worker，互不抢卡，天然“单卡串行，多卡并行”
    all_failed = []
    with ThreadPoolExecutor(max_workers=ng) as ex:
        futures = []
        for gid, jobs in zip(gpu_ids, buckets):
            futures.append(ex.submit(run_one_on_gpu, gid, jobs))
        for fut in futures:
            all_failed.extend(fut.result())

    print("全部完成 ✅")
    if all_failed:
        print(f"共有 {len(all_failed)} 个任务失败：")
        for p in all_failed:
            print("  -", p)

if __name__ == "__main__":
    main()
