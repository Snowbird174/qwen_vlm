"""演示界面的单请求锁。

同一时刻只允许一个推理在途。拿不到锁的新请求直接提示正忙，不排队。
"""

from __future__ import annotations

import threading


class BusyError(RuntimeError):
    """已经有一个推理请求在跑。"""


class InferenceGate:
    def __init__(self) -> None:
        self._lock = threading.Lock()

    def try_acquire(self) -> bool:
        return self._lock.acquire(blocking=False)

    def release(self) -> None:
        self._lock.release()

    def __enter__(self) -> InferenceGate:
        if not self.try_acquire():
            raise BusyError("正忙：上一个推理还没结束")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
