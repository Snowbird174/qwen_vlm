"""标注工具的数据层：读写一图一行的标注 CSV（列见 convert.ANNOTATION_COLUMNS）。

图片按文件名排序（采集工具的文件名带时间戳，等于按拍摄顺序）。
variant_id 由图片序号生成（v001、v002……），因此同一 session 内唯一且稳定。
废片记在 ``<csv>.skipped.json``，不写进 CSV。
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from qwen_vlm import DATA_DIR
from qwen_vlm.common.categories import CATEGORY_IDS, NAME_ZH
from qwen_vlm.data.convert import ANNOTATION_COLUMNS, ConversionError, csv_to_samples, load_question_templates
from qwen_vlm.data.validate import validate_samples

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
TEMPLATES = Path(__file__).resolve().parents[3] / "configs/data/question_templates.yaml"


def default_session_id(raw_dir: Path) -> str:
    return re.sub(r"[^0-9A-Za-z]", "", raw_dir.name.replace("session", "s")) or "s001"


class LabelStore:
    def __init__(self, raw_dir: Path | str, session_id: str, csv_path: Path | str, data_dir: Path | str = DATA_DIR):
        if "_" in session_id:
            raise ValueError("session_id 不能包含下划线")
        self.data_dir = Path(data_dir).resolve()
        self.raw_dir = Path(raw_dir).resolve()
        self.session_id = session_id
        self.csv_path = Path(csv_path)
        self.skip_path = self.csv_path.with_suffix(".skipped.json")
        self.images = sorted(p for p in self.raw_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if not self.images:
            raise ValueError(f"{self.raw_dir} 下没有图片")
        self.rel_paths = [p.relative_to(self.data_dir).as_posix() for p in self.images]
        self.rows: dict[str, dict] = {}
        self.skipped: set[str] = set()
        self._load()

    # ---------- 读写 ----------
    def _load(self) -> None:
        if self.csv_path.exists():
            with self.csv_path.open(encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    if row.get("image_path") in self.rel_paths:
                        self.rows[row["image_path"]] = {c: row.get(c, "") for c in ANNOTATION_COLUMNS}
        if self.skip_path.exists():
            self.skipped = set(json.loads(self.skip_path.read_text(encoding="utf-8")))

    def save(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ANNOTATION_COLUMNS)
            writer.writeheader()
            for rel in self.rel_paths:
                if rel in self.rows and rel not in self.skipped:
                    writer.writerow(self.rows[rel])
        self.skip_path.write_text(json.dumps(sorted(self.skipped), ensure_ascii=False, indent=1), encoding="utf-8")

    # ---------- 查询 ----------
    def __len__(self) -> int:
        return len(self.images)

    def row(self, i: int) -> dict | None:
        return self.rows.get(self.rel_paths[i])

    def status(self, i: int) -> str:
        rel = self.rel_paths[i]
        return "废片" if rel in self.skipped else ("已标" if rel in self.rows else "未标")

    def n_done(self) -> int:
        return sum(1 for r in self.rel_paths if r in self.rows or r in self.skipped)

    def first_unlabeled(self) -> int:
        for i, rel in enumerate(self.rel_paths):
            if rel not in self.rows and rel not in self.skipped:
                return i
        return 0

    def previous_labeled(self, i: int) -> dict | None:
        for j in range(i - 1, -1, -1):
            rel = self.rel_paths[j]
            if rel in self.rows and rel not in self.skipped:
                return self.rows[rel]
        return None

    def next_layout_id(self) -> str:
        nums = [int(m.group(1)) for r in self.rows.values() if (m := re.fullmatch(r"l(\d+)", r["layout_id"]))]
        return f"l{(max(nums) + 1) if nums else 1:03d}"

    # ---------- 修改 ----------
    def skip(self, i: int) -> None:
        rel = self.rel_paths[i]
        self._clear_pair(rel)
        self.rows.pop(rel, None)
        self.skipped.add(rel)
        self.save()

    def set(
        self,
        i: int,
        *,
        layout_id: str,
        states: dict[str, str],
        unknown_reason: str = "",
        condition_tags: list[str] | None = None,
        hard_negative_tags: list[str] | None = None,
        pair_with_previous: bool = False,
        verified: bool = True,
        notes: str = "",
    ) -> list[str]:
        """保存第 i 张的标注，返回提示信息。数据不合法时抛 ValueError，不写入。"""
        layout_id = layout_id.strip()
        if not re.fullmatch(r"[0-9A-Za-z]+", layout_id):
            raise ValueError("摆放编号只能用字母和数字，例如 l001")
        if set(states) != set(CATEGORY_IDS) or any(v not in ("present", "absent", "unknown") for v in states.values()):
            raise ValueError("5 件器材都要选状态")
        unknown = [c for c in CATEGORY_IDS if states[c] == "unknown"]
        reason = re.sub(r"[=;]", "，", unknown_reason.strip())
        if unknown and not reason:
            raise ValueError(f"{'、'.join(NAME_ZH[c] for c in unknown)}选了看不清，请填写原因（如：被手挡住）")

        rel = self.rel_paths[i]
        messages: list[str] = []
        old_pair = (self.rows.get(rel) or {}).get("pair_group_id", "")
        row = {
            "session_id": self.session_id,
            "layout_id": layout_id,
            "variant_id": f"v{i + 1:03d}",
            "image_path": rel,
            "pair_group_id": "",
            **{c: states[c] for c in CATEGORY_IDS},
            "unknown_reason": ";".join(f"{c}={reason}" for c in unknown),
            "condition_tags": "|".join(condition_tags or []),
            "hard_negative_tags": "|".join(hard_negative_tags or []),
            "annotation_verified": "true" if verified else "false",
            "notes": notes.strip(),
        }

        if pair_with_previous:
            row["pair_group_id"] = self._pair_id_with_previous(i, row, old_pair)
        elif old_pair:
            self._clear_pair(rel)
            messages.append("已取消与上一张的配对")

        self.skipped.discard(rel)
        self.rows[rel] = row
        if row["pair_group_id"]:
            self.rows[self.rel_paths[i - 1]]["pair_group_id"] = row["pair_group_id"]
            messages.append(f"已与上一张配对：{row['pair_group_id']}")
        self.save()
        return messages

    def _pair_id_with_previous(self, i: int, row: dict, old_pair: str) -> str:
        if i == 0:
            raise ValueError("第一张没有上一张，不能配对")
        prev_rel = self.rel_paths[i - 1]
        prev = self.rows.get(prev_rel)
        if prev is None or prev_rel in self.skipped:
            raise ValueError("上一张还没标注（或是废片），不能配对")
        if prev["layout_id"] != row["layout_id"]:
            raise ValueError("配对的两张必须是同一个摆放编号")
        diff = [c for c in CATEGORY_IDS if prev[c] != row[c]]
        if len(diff) != 1:
            names = "、".join(NAME_ZH[c] for c in diff) or "没有"
            raise ValueError(
                f"配对的两张必须恰好只有 1 件器材状态不同，这张和上一张不同的有 {len(diff)} 件（{names}）。"
                "请取消勾选「与上一张配对」再保存"
            )
        if prev["pair_group_id"] and prev["pair_group_id"] != old_pair:
            raise ValueError("上一张已经和更早的一张配过对了，一张图只能属于一对")
        if old_pair and prev["pair_group_id"] == old_pair:
            return old_pair
        nums = [
            int(m.group(1))
            for r in self.rows.values()
            if r["layout_id"] == row["layout_id"] and (m := re.search(r"_p(\d+)$", r["pair_group_id"]))
        ]
        return f"{self.session_id}_{row['layout_id']}_p{(max(nums) + 1) if nums else 1:02d}"

    def _clear_pair(self, rel: str) -> None:
        pair = (self.rows.get(rel) or {}).get("pair_group_id", "")
        if pair:
            for r in self.rows.values():
                if r["pair_group_id"] == pair:
                    r["pair_group_id"] = ""

    # ---------- 检查 ----------
    def check(self) -> tuple[list[str], int]:
        """用正式的转换和校验流程检查当前 CSV，返回 (错误列表, 样本数)。"""
        self.save()
        if not self.rows:
            return ["还没有任何标注"], 0
        try:
            samples = csv_to_samples(self.csv_path, load_question_templates(TEMPLATES), source="real_lab", seed=0)
        except ConversionError as exc:
            return exc.errors, 0
        return validate_samples(samples, data_dir=self.data_dir, check_images=True), len(samples)
