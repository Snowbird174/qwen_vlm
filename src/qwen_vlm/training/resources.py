"""磁盘和 GPU 占用检查。只等待，不结束别人的进程。"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

MIN_FREE_BYTES = 5 * 1024**3
OWN_GPU_CAP_BYTES = 20 * 1024**3


def free_bytes(path: str | Path = "/root/gpufree-data") -> int:
    return int(shutil.disk_usage(path).free)


def disk_guard(path: str | Path = "/root/gpufree-data", min_free: int = MIN_FREE_BYTES) -> int:
    free = free_bytes(path)
    gib = free / (1024**3)
    print(f"磁盘剩余 {gib:.2f} GiB ({path})")
    if free < min_free:
        raise SystemExit(f"磁盘剩余 {gib:.2f} GiB，低于 5 GiB，停止本轮训练侧写入。")
    return free


def query_gpu() -> tuple[int, int, int]:
    """返回 (free_mib, used_mib, total_mib)。"""
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.free,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip().splitlines()[0]
    free_mib, used_mib, total_mib = [int(part.strip()) for part in raw.split(",")]
    return free_mib, used_mib, total_mib


def wait_for_gpu(min_free_mib: int = 18432, poll_s: int = 30, timeout_s: int = 3600) -> tuple[int, int, int]:
    """空闲显存不够就等待。不会 kill 其他进程。"""
    started = time.time()
    while True:
        free_mib, used_mib, total_mib = query_gpu()
        print(f"GPU 显存 空闲 {free_mib} MiB / 已用 {used_mib} MiB / 总共 {total_mib} MiB")
        if free_mib >= min_free_mib:
            return free_mib, used_mib, total_mib
        if time.time() - started > timeout_s:
            raise SystemExit(f"等待 GPU 空闲超过 {timeout_s} 秒，当前空闲 {free_mib} MiB，停止。")
        print(f"空闲显存低于 {min_free_mib} MiB，{poll_s} 秒后再看。")
        time.sleep(poll_s)


def cap_process_memory(total_mib: int, free_mib: int) -> float:
    """把本进程可申请的显存限制在 20 GiB 以内，并且不超过当前空闲。"""
    import torch

    total_bytes = torch.cuda.get_device_properties(0).total_memory
    free_bytes_now = free_mib * 1024**2
    cap = min(OWN_GPU_CAP_BYTES, max(1, free_bytes_now - 1024**3))
    fraction = min(0.95, cap / total_bytes)
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    print(f"本进程显存上限约 {cap / (1024**3):.2f} GiB（fraction={fraction:.3f}，卡总显存 {total_mib} MiB）")
    return fraction
