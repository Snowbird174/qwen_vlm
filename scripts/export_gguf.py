#!/usr/bin/env python3
"""把 Hugging Face 格式的 Qwen3-VL 转成 GGUF，并写出量化文件和 manifest。

语言权重：F16，再由 llama-quantize 得到 Q8_0 与 Q4_K_M。
视觉 mmproj：单独一份 F16，不参与 llama-quantize。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def disk_free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def require_free_gb(path: Path, minimum_gb: float, what: str) -> None:
    free = disk_free_bytes(path)
    print(f"磁盘剩余 {free / 1024**3:.2f} GiB（{what} 之前）", flush=True)
    if free < minimum_gb * 1024**3:
        raise SystemExit(f"剩余不足 {minimum_gb} GiB，停止：{what}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_revision(hf_dir: Path) -> str | None:
    env_rev = os.environ.get("HF_REVISION")
    if env_rev:
        return env_rev
    candidates = [hf_dir / "manifest.json", hf_dir.parent / "manifest.json"]
    for path in candidates:
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("revision"):
            return str(data["revision"])
    return None


def _jsonify(value: object) -> object:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "tolist"):
        return _jsonify(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def gguf_metadata(path: Path, gguf_py: Path) -> dict[str, object]:
    sys.path.insert(0, str(gguf_py))
    import gguf  # noqa: E402

    reader = gguf.GGUFReader(str(path))
    fields: dict[str, object] = {}
    for key in (
        "general.architecture",
        "general.name",
        "general.file_type",
        "qwen3vl.n_deepstack_layers",
        "tokenizer.chat_template",
        "clip.projector_type",
        "clip.vision.is_deepstack_layers",
        "clip.vision.patch_size",
        "clip.vision.spatial_merge_size",
    ):
        field = reader.get_field(key)
        if field is None:
            fields[key] = None
            continue
        value = _jsonify(field.contents())
        if key == "tokenizer.chat_template" and isinstance(value, str):
            fields[key] = {"present": True, "chars": len(value)}
        else:
            fields[key] = value
    return fields


def run(cmd: list[str], env: dict[str, str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env, cwd=cwd)


def main() -> None:
    parser = argparse.ArgumentParser(description="HF Qwen3-VL → F16/Q8_0/Q4_K_M GGUF + mmproj")
    parser.add_argument("hf_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("tag")
    parser.add_argument("--python", type=Path, default=None)
    parser.add_argument("--quantize-bin", type=Path, default=None)
    parser.add_argument(
        "--delete-f16",
        action="store_true",
        help="量化完成后立刻删除语言 F16。默认保留，等冒烟对比结束再删。",
    )
    args = parser.parse_args()

    root = project_root()
    hf_dir = args.hf_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    llama_root = root / "vendor" / "llama.cpp"
    convert = llama_root / "convert_hf_to_gguf.py"
    quantize = args.quantize_bin or (llama_root / "build" / "bin" / "llama-quantize")
    py = args.python or (root / "envs" / "qwen_vlm" / "bin" / "python")
    if not convert.is_file():
        raise SystemExit(f"缺少转换脚本 {convert}")
    if not Path(quantize).is_file():
        raise SystemExit(f"缺少 llama-quantize：{quantize}。先运行 scripts/export_build_llamacpp_server.sh")
    if not hf_dir.is_dir():
        raise SystemExit(f"HF 目录不存在：{hf_dir}")

    require_free_gb(out_dir, 5, "开始转换")

    dep_dir = root / "work" / "deploy_pydeps"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = str(dep_dir) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    # set_vocab 会先 import sentencepiece，缺文件时才退回 GPT-2 tokenizer。转换进程需要这个包，但不改共享环境。
    probe = subprocess.run([str(py), "-c", "import sentencepiece"], env=env, capture_output=True, text=True)
    if probe.returncode != 0:
        dep_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [str(py), "-m", "pip", "install", "--target", str(dep_dir), "sentencepiece"],
            check=True,
            env={k: v for k, v in env.items() if k.lower() not in {"http_proxy", "https_proxy", "all_proxy"}},
        )

    lang_f16 = out_dir / f"{args.tag}-f16.gguf"
    mmproj = out_dir / f"mmproj-{args.tag}-f16.gguf"
    lang_q8 = out_dir / f"{args.tag}-q8_0.gguf"
    lang_q4 = out_dir / f"{args.tag}-q4_k_m.gguf"

    run(
        [str(py), str(convert), str(hf_dir), "--outtype", "f16", "--outfile", str(lang_f16)],
        env,
        cwd=llama_root,
    )
    require_free_gb(out_dir, 5, "语言 F16 已写出，准备 mmproj")
    run(
        [str(py), str(convert), str(hf_dir), "--outtype", "f16", "--mmproj", "--outfile", str(mmproj)],
        env,
        cwd=llama_root,
    )
    if not lang_f16.is_file() or not mmproj.is_file():
        raise SystemExit(f"转换未生成预期文件：{lang_f16} / {mmproj}")

    require_free_gb(out_dir, 5, "准备量化")
    run([str(quantize), str(lang_f16), str(lang_q8), "Q8_0"], env)
    require_free_gb(out_dir, 5, "Q8_0 已写出，准备 Q4_K_M")
    run([str(quantize), str(lang_f16), str(lang_q4), "Q4_K_M"], env)

    files = [
        (lang_f16, "language", "F16", "语言模型 float16。这不是整模的精度。"),
        (lang_q8, "language", "Q8_0", "只量化语言权重。mmproj 仍是 F16。"),
        (lang_q4, "language", "Q4_K_M", "只量化语言权重。mmproj 仍是 F16，不能把整个 VLM 说成 INT4。"),
        (mmproj, "mmproj", "F16", "视觉编码器、merger 和 DeepStack 投影，保持 F16，未经过 llama-quantize。"),
    ]
    manifest_files = []
    for path, role, quant, note in files:
        meta = gguf_metadata(path, llama_root / "gguf-py")
        manifest_files.append(
            {
                "name": path.name,
                "role": role,
                "quant": quant,
                "note": note,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "gguf_metadata": meta,
            }
        )

    version_text = (root / "deploy" / "jetson" / "LLAMACPP_VERSION.md").read_text(encoding="utf-8")
    commit = None
    tag = None
    for line in version_text.splitlines():
        if line.startswith("- commit:"):
            commit = line.split("`", 2)[1]
        if line.startswith("- tag:"):
            tag = line.split("`", 2)[1]

    py_ver = subprocess.check_output([str(py), "-c", "import sys,torch,transformers; print(sys.version.split()[0]); print(torch.__version__); print(transformers.__version__)"], text=True).splitlines()
    manifest = {
        "tag": args.tag,
        "hf_model_dir": str(hf_dir),
        "hf_revision": read_revision(hf_dir),
        "llamacpp_tag": tag,
        "llamacpp_commit": commit,
        "conversion": {
            "python": str(py),
            "python_version": py_ver[0],
            "torch": py_ver[1],
            "transformers": py_ver[2],
            "note": (
                "转换使用 envs/qwen_vlm（已有 torch 2.11 CUDA 与 transformers 5.5.4），"
                "进程内 CUDA_VISIBLE_DEVICES 为空，权重在 CPU 上读取。"
                "没有把 CPU 版 torch 和 llama.cpp 要求的 transformers==4.57.6 装进 envs/gguf，"
                "因为那一套依赖大约 1.5–2GB，会挤占 GGUF 的磁盘预算。"
                "sentencepiece 若缺失，只装到 work/deploy_pydeps，不改共享环境。"
            ),
        },
        "precision": {
            "language_f16": "F16",
            "language_q8_0": "Q8_0",
            "language_q4_k_m": "Q4_K_M",
            "mmproj": "F16",
            "statement": "mmproj 视觉部分是 F16；Q4_K_M / Q8_0 只作用于语言 GGUF。",
        },
        "files": manifest_files,
        "disk_free_bytes_after_quantize": disk_free_bytes(out_dir),
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"写出 {manifest_path}", flush=True)

    if args.delete_f16:
        lang_f16.unlink()
        manifest["language_f16_deleted_after_quantize"] = True
        print(f"已删除语言 F16：{lang_f16.name}。mmproj 仍保留。", flush=True)
    else:
        manifest["language_f16_deleted_after_quantize"] = False
    manifest["disk_free_bytes_final"] = disk_free_bytes(out_dir)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
