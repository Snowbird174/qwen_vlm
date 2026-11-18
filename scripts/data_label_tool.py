"""启动网页标注工具。

例：
  python scripts/data_label_tool.py data/raw/session_001
然后在浏览器打开 http://localhost:7861 。标注随每次保存写入 data/annotations/<session_id>.csv，可随时关掉再继续。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from qwen_vlm import PROJECT_ROOT
from qwen_vlm.data.label_app import build_app
from qwen_vlm.data.label_store import LabelStore, default_session_id


def main() -> None:
    ap = argparse.ArgumentParser(description="器材点检网页标注工具")
    ap.add_argument("raw_dir", help="一个批次的照片目录，例如 data/raw/session_001")
    ap.add_argument("--session-id", default=None, help="写进标注表的批次编号（不能含下划线）；默认由目录名生成，如 session_001 -> s001")
    ap.add_argument("--csv", default=None, help="标注表路径；默认 data/annotations/<session_id>.csv")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7861)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    if not raw_dir.is_absolute():
        raw_dir = PROJECT_ROOT / raw_dir
    session_id = args.session_id or default_session_id(raw_dir)
    csv_path = Path(args.csv) if args.csv else PROJECT_ROOT / "data/annotations" / f"{session_id}.csv"
    store = LabelStore(raw_dir, session_id, csv_path)
    print(f"批次 {session_id}：{len(store)} 张照片，已完成 {store.n_done()} 张；标注表 {csv_path}")
    build_app(store).launch(server_name=args.host, server_port=args.port, allowed_paths=[str(raw_dir)])


if __name__ == "__main__":
    main()
