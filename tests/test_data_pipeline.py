"""数据校验、划分、泄漏和 CSV 展开的反例。这些测试应当在数据出错时失败。"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from qwen_vlm import PROJECT_ROOT
from qwen_vlm.common.categories import CATEGORY_IDS
from qwen_vlm.data.convert import ConversionError, csv_to_samples, load_question_templates
from qwen_vlm.data.leakage import check_leakage
from qwen_vlm.data.split import assign_splits, compute_split_counts
from qwen_vlm.data.validate import validate_samples

TEMPLATES = PROJECT_ROOT / "configs" / "data" / "question_templates.yaml"


def _states(**overrides: str) -> dict[str, str]:
    base = {cat: "present" for cat in CATEGORY_IDS}
    base.update(overrides)
    return base


def _write_png(path: Path, color: tuple[int, int, int] = (20, 40, 60)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 48), color).save(path)


def _sample(rel: str = "a.png", **overrides) -> dict:
    sample = {
        "sample_id": "s01_l001_v01_q01",
        "image_path": rel,
        "session_id": "s01",
        "scene_group_id": "s01_l001",
        "pair_group_id": None,
        "task": "checklist",
        "required_items": ["screwdriver", "pliers"],
        "question_zh": "所需器材都在吗？",
        "item_states": _states(),
        "condition_tags": ["baseline"],
        "hard_negative_tags": [],
        "unknown_reason": {},
        "source": "synthetic_smoke",
        "annotation_verified": False,
    }
    sample.update(overrides)
    return sample


def _layout_samples(session: str, layout: str, n: int = 1) -> list[dict]:
    rows = []
    for index in range(n):
        variant = f"v{index + 1:02d}"
        rows.append(
            _sample(
                f"{session}_{layout}_{variant}.png",
                sample_id=f"{session}_{layout}_{variant}_q01",
                session_id=session,
                scene_group_id=f"{session}_{layout}",
            )
        )
    return rows


def test_unknown_reason_mismatch_is_reported(tmp_path: Path):
    _write_png(tmp_path / "a.png")
    sample = _sample(item_states=_states(multimeter="unknown"), unknown_reason={})
    errors = validate_samples([sample], data_dir=tmp_path)
    assert any("unknown_reason" in msg for msg in errors)


def test_duplicate_sample_id_is_reported(tmp_path: Path):
    _write_png(tmp_path / "a.png")
    errors = validate_samples([_sample(), _sample()], data_dir=tmp_path)
    assert any("sample_id 重复" in msg for msg in errors)


def test_missing_image_is_reported(tmp_path: Path):
    errors = validate_samples([_sample("missing.png")], data_dir=tmp_path)
    assert any("图片不存在" in msg for msg in errors)


def test_pair_with_two_changed_categories_is_reported(tmp_path: Path):
    _write_png(tmp_path / "a.png", (200, 20, 20))
    _write_png(tmp_path / "b.png", (20, 20, 200))
    left = _sample("a.png", sample_id="s01_l001_v01_q01", pair_group_id="p01")
    right = _sample(
        "b.png",
        sample_id="s01_l001_v02_q01",
        pair_group_id="p01",
        item_states=_states(pliers="absent", hammer="absent"),
    )
    errors = validate_samples([left, right], data_dir=tmp_path)
    assert any("pair_group_id=p01" in msg and "实际 2 个" in msg for msg in errors)

    right["item_states"] = _states(pliers="absent")
    assert validate_samples([left, right], data_dir=tmp_path) == []


def test_pair_with_three_images_is_reported(tmp_path: Path):
    rows = []
    for name, cat in (("a", None), ("b", "pliers"), ("c", "hammer")):
        _write_png(tmp_path / f"{name}.png", (40, 80, 120))
        states = _states()
        if cat:
            states[cat] = "absent"
        rows.append(
            _sample(f"{name}.png", sample_id=f"s01_l001_{name}_q01", pair_group_id="p01", item_states=states)
        )
    errors = validate_samples(rows, data_dir=tmp_path)
    assert any("期望恰好 2 张不同的图，实际 3 张" in msg for msg in errors)


def test_presence_state_must_match_item_states(tmp_path: Path):
    _write_png(tmp_path / "a.png")
    checklist = _sample(item_states=_states(pliers="present"))
    presence = _sample(
        sample_id="s01_l001_v01_q02",
        task="presence",
        target_item="pliers",
        required_items=["pliers"],
        item_states=_states(pliers="absent"),
    )
    errors = validate_samples([checklist, presence], data_dir=tmp_path)
    assert any("item_states 不一致" in msg for msg in errors)
    assert any("target_item=pliers" in msg and "不一致" in msg for msg in errors)


def test_scene_group_is_not_split_across_buckets():
    samples = []
    for index in range(9):
        samples.extend(_layout_samples("s01", f"l{index:03d}"))
    buckets, _ = assign_splits(samples, seed=0)
    seen: dict[str, str] = {}
    for name, rows in buckets.items():
        for row in rows:
            previous = seen.setdefault(row["scene_group_id"], name)
            assert previous == name
    assert len(buckets["train"]) == 6
    assert len(buckets["val"]) == 2
    assert len(buckets["test"]) == 1
    again, _ = assign_splits(samples, seed=0)
    assert [row["sample_id"] for row in buckets["train"]] == [row["sample_id"] for row in again["train"]]


def test_variants_and_cross_scene_pairs_stay_together():
    samples = _layout_samples("s01", "g1", n=2)
    samples[0]["pair_group_id"] = "pair_cross"
    samples.append(
        _sample(
            "s01_g2_v01.png",
            sample_id="s01_g2_v01_q01",
            scene_group_id="s01_g2",
            pair_group_id="pair_cross",
            item_states=_states(hammer="absent"),
        )
    )
    for layout in ("g3", "g4", "g5"):
        samples.extend(_layout_samples("s01", layout))
    buckets, _ = assign_splits(samples, seed=0)
    where = {row["sample_id"]: name for name, rows in buckets.items() for row in rows}
    assert where["s01_g1_v01_q01"] == where["s01_g1_v02_q01"] == where["s01_g2_v01_q01"]


def test_holdout_session_goes_to_test_ood_only():
    samples = []
    for layout in ("l001", "l002", "l003"):
        samples.extend(_layout_samples("s01", layout))
    for layout in ("l001", "l002"):
        samples.extend(_layout_samples("s02", layout))
    buckets, _ = assign_splits(samples, seed=0, holdout_sessions=["s02"])
    ood = {row["sample_id"] for row in buckets["test_ood"]}
    assert ood == {row["sample_id"] for row in samples if row["session_id"] == "s02"}
    for name in ("train", "val", "test"):
        assert all(row["session_id"] != "s02" for row in buckets[name])
    assert sum(len(rows) for rows in buckets.values()) == len(samples)


def test_split_counts_cover_small_sets():
    assert compute_split_counts(9, (0.7, 0.15, 0.15)) == [6, 2, 1]
    assert compute_split_counts(6, (0.7, 0.15, 0.15)) == [4, 1, 1]
    assert compute_split_counts(3, (0.7, 0.15, 0.15)) == [1, 1, 1]
    assert sum(compute_split_counts(10, (0.7, 0.15, 0.15))) == 10


def test_cross_split_scene_leak_is_detected(tmp_path: Path):
    _write_png(tmp_path / "a.png", (180, 40, 40))
    _write_png(tmp_path / "b.png", (40, 40, 180))
    train = _sample("a.png", scene_group_id="s01_l001")
    val = _sample("b.png", sample_id="s01_l001_v02_q01", scene_group_id="s01_l001")
    report = check_leakage({"train": [train], "val": [val]}, data_dir=tmp_path)
    assert report["scene_group_leaks"]
    assert report["scene_group_leaks"][0]["scene_group_id"] == "s01_l001"
    assert report["ok"] is False


def test_cross_split_pair_leak_is_detected(tmp_path: Path):
    _write_png(tmp_path / "a.png", (10, 180, 40))
    _write_png(tmp_path / "b.png", (180, 180, 20))
    train = _sample("a.png", scene_group_id="s01_l001", pair_group_id="p01")
    val = _sample(
        "b.png",
        sample_id="s02_l009_v01_q01",
        session_id="s02",
        scene_group_id="s02_l009",
        pair_group_id="p01",
        item_states=_states(screwdriver="absent"),
    )
    report = check_leakage({"train": [train], "val": [val]}, data_dir=tmp_path, phash_threshold=0)
    assert any(item["pair_group_id"] == "p01" for item in report["pair_group_leaks"])
    assert report["ok"] is False


def test_exact_duplicate_across_splits_is_detected(tmp_path: Path):
    _write_png(tmp_path / "a.png", (12, 80, 140))
    _write_png(tmp_path / "b.png", (12, 80, 140))
    train = _sample("a.png", scene_group_id="s01_l001")
    val = _sample("b.png", sample_id="s01_l002_v01_q01", scene_group_id="s01_l002")
    report = check_leakage({"train": [train], "val": [val]}, data_dir=tmp_path)
    assert any(item["cross_split"] for item in report["exact_duplicates"])
    assert report["near_duplicates"]
    assert report["ok"] is False


def test_csv_expansion_uses_default_questions_and_is_deterministic(tmp_path: Path):
    csv_path = tmp_path / "ann.csv"
    csv_path.write_text(
        "session_id,layout_id,variant_id,image_path,pair_group_id,screwdriver,pliers,cable_bundle,multimeter,hammer,unknown_reason,condition_tags,hard_negative_tags,annotation_verified,notes\n"
        "s01,l001,v01,raw/s01/l001/v01.jpg,s01_l001_p01,present,absent,present,unknown,present,"
        "multimeter=被纸挡住机身，无法确认,lighting_warm|baseline,,false,给人看的备注\n"
        "s01,l001,v02,raw/s01/l001/v02.jpg,s01_l001_p01,present,absent,present,unknown,absent,"
        "multimeter=被纸挡住机身，无法确认,lighting_warm,,false,相对 v01 只拿走锤子\n",
        encoding="utf-8",
    )
    templates = load_question_templates(TEMPLATES)
    kwargs = dict(question_set="default", checklists_per_image=2, presence_per_image=1, seed=0, source="real_lab")
    first = csv_to_samples(csv_path, templates, **kwargs)
    second = csv_to_samples(csv_path, templates, **kwargs)
    assert first == second
    v01 = [row for row in first if row["image_path"].endswith("v01.jpg")]
    assert [row["sample_id"] for row in v01] == ["s01_l001_v01_q01", "s01_l001_v01_q02", "s01_l001_v01_q03"]
    assert all(row["scene_group_id"] == "s01_l001" for row in first)
    assert all(row["unknown_reason"] == {"multimeter": "被纸挡住机身，无法确认"} for row in first)
    assert all("notes" not in row for row in first)
    pools = {tuple(pool) for pool in templates["question_sets"]["default"]["checklist_pools"]}
    default_questions = set(templates["question_sets"]["default"]["checklist_questions"])
    checklist = [row for row in v01 if row["task"] == "checklist"]
    presence = [row for row in v01 if row["task"] == "presence"]
    assert len(checklist) == 2 and len(presence) == 1
    for row in checklist:
        assert tuple(row["required_items"]) in pools
        assert 2 <= len(row["required_items"]) <= 4
        assert row["question_zh"] in default_questions
        assert not row["question_zh"].startswith("【测试问法】")
    assert presence[0]["target_item"] in CATEGORY_IDS
    assert presence[0]["item_states"] == checklist[0]["item_states"]
    assert presence[0]["required_items"] == [presence[0]["target_item"]]
    assert validate_samples(first, check_images=False) == []

    held_out = csv_to_samples(
        csv_path, templates, question_set="test_only", checklists_per_image=1, presence_per_image=0, seed=0, source="real_lab"
    )
    assert held_out[0]["question_zh"].startswith("【测试问法】")


def test_csv_rejects_unknown_reason_that_does_not_match(tmp_path: Path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text(
        "session_id,layout_id,variant_id,image_path,pair_group_id,screwdriver,pliers,cable_bundle,multimeter,hammer,unknown_reason,condition_tags,hard_negative_tags,annotation_verified,notes\n"
        "s01,l001,v01,raw/s01/l001/v01.jpg,,present,present,present,unknown,present,{},,,false,\n",
        encoding="utf-8",
    )
    templates = load_question_templates(TEMPLATES)
    try:
        csv_to_samples(csv_path, templates, source="real_lab")
    except ConversionError as exc:
        assert any("unknown_reason" in msg for msg in exc.errors)
    else:
        raise AssertionError("不匹配的 unknown_reason 应该被拒绝")
