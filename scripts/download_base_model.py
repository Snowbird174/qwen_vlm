"""Fetch the pinned Qwen3-VL-2B-Instruct revision into artifacts/base/ and verify it.

Every file is checked against the Hugging Face file tree of the pinned revision
(git blob SHA-1 for small files, LFS SHA-256 for weights), so files may come from any
mirror. On this server HF's CDN is ~30 KB/s while ModelScope (via the local proxy) is
2-6 MB/s, hence the default weights source.

Usage (after `source env/activate.sh`):
    python scripts/download_base_model.py            # download missing files, verify, write manifest
    python scripts/download_base_model.py --verify-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

REPO_ID = "Qwen/Qwen3-VL-2B-Instruct"
REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
HF_API = "https://hf-mirror.com/api/models/{repo}/tree/{rev}"
HF_FILE = "https://hf-mirror.com/{repo}/resolve/{rev}/{name}"
MS_FILE = "https://modelscope.cn/models/{repo}/resolve/master/{name}"
PROXY = os.environ.get("QWEN_VLM_DL_PROXY", "http://127.0.0.1:7897")


def _hash(path: Path, algo: str, git_blob: bool = False) -> str:
    h = hashlib.new(algo)
    if git_blob:
        h.update(f"blob {path.stat().st_size}\0".encode())
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _no_proxy_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def fetch_tree(cache: Path) -> list[dict]:
    if not cache.exists():
        url = HF_API.format(repo=REPO_ID, rev=REVISION)
        cache.write_bytes(_no_proxy_opener().open(url, timeout=60).read())
    return [e for e in json.loads(cache.read_text()) if e["type"] == "file"]


def verify(path: Path, entry: dict) -> bool:
    if not path.exists() or path.stat().st_size != entry["size"]:
        return False
    if entry.get("lfs"):
        return _hash(path, "sha256") == entry["lfs"]["oid"]
    return _hash(path, "sha1", git_blob=True) == entry["oid"]


def download(entry: dict, dest: Path) -> None:
    name = entry["path"]
    url = (MS_FILE if entry.get("lfs") else HF_FILE).format(repo=REPO_ID, rev=REVISION, name=name)
    cmd = ["aria2c", "-x", "8", "-s", "8", "-k", "20M", "--max-tries=0", "--retry-wait=5", "--timeout=60",
           "--file-allocation=none", "--continue=true", "-d", str(dest.parent), "-o", dest.name, url]
    if entry.get("lfs"):
        cmd.insert(1, f"--all-proxy={PROXY}")
    env = {k: v for k, v in os.environ.items() if not k.lower().endswith("_proxy")}
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    root = Path(os.environ["QWEN_VLM_ROOT"])
    model_dir = root / "artifacts/base/Qwen3-VL-2B-Instruct"
    model_dir.mkdir(parents=True, exist_ok=True)
    entries = fetch_tree(root / "artifacts/base/hf_tree.json")

    files, failed = [], []
    for e in entries:
        if e["path"] == ".gitattributes":
            continue
        path = model_dir / e["path"]
        ok = verify(path, e)
        if not ok and not args.verify_only:
            download(e, path)
            ok = verify(path, e)
        if not ok:
            failed.append(e["path"])
            continue
        files.append({
            "name": e["path"],
            "bytes": e["size"],
            "sha256": _hash(path, "sha256"),
            "verified_against": "hf_lfs_sha256" if e.get("lfs") else "hf_git_blob_sha1",
        })
    if failed:
        raise SystemExit(f"verification failed: {failed}")

    manifest = {
        "repo_id": REPO_ID,
        "revision": REVISION,
        "local_path": str(model_dir),
        "weights_source": "modelscope.cn (Qwen/Qwen3-VL-2B-Instruct), verified against HF LFS sha256",
        "small_files_source": "hf-mirror.com at the pinned revision",
        "verified_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "total_bytes": sum(f["bytes"] for f in files),
        "disk_free_bytes_after": shutil.disk_usage(root).free,
        "files": files,
    }
    (root / "artifacts/base/manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print("OK", len(files), "files verified ->", root / "artifacts/base/manifest.json")


if __name__ == "__main__":
    main()
