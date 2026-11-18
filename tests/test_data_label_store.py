import pytest
from PIL import Image

from qwen_vlm.data.label_store import LabelStore, default_session_id

ALL = {"screwdriver": "present", "pliers": "present", "cable_bundle": "present", "multimeter": "present", "hammer": "present"}


@pytest.fixture
def store(tmp_path):
    raw = tmp_path / "data/raw/session_001"
    raw.mkdir(parents=True)
    for k in range(4):
        Image.new("RGB", (64, 48), (40 * k, 100, 100)).save(raw / f"session_001_2026_{k}.png")
    return LabelStore(raw, "s001", tmp_path / "data/annotations/s001.csv", data_dir=tmp_path / "data")


def test_default_session_id(tmp_path):
    assert default_session_id(tmp_path / "session_001") == "s001"


def test_pair_and_check_pass(store):
    store.set(0, layout_id="l001", states=ALL)
    msgs = store.set(1, layout_id="l001", states={**ALL, "pliers": "absent"}, pair_with_previous=True)
    assert "s001_l001_p01" in msgs[0]
    assert store.row(0)["pair_group_id"] == store.row(1)["pair_group_id"] == "s001_l001_p01"
    store.set(2, layout_id="l002", states={**ALL, "hammer": "unknown"}, unknown_reason="被手挡住")
    assert store.row(2)["unknown_reason"] == "hammer=被手挡住"
    store.skip(3)
    errors, n = store.check()
    assert errors == [] and n > 0


def test_pair_rejects_two_differences_and_other_layout(store):
    store.set(0, layout_id="l001", states=ALL)
    with pytest.raises(ValueError, match="恰好只有 1 件"):
        store.set(1, layout_id="l001", states={**ALL, "pliers": "absent", "hammer": "absent"}, pair_with_previous=True)
    with pytest.raises(ValueError, match="同一个摆放"):
        store.set(1, layout_id="l002", states={**ALL, "pliers": "absent"}, pair_with_previous=True)
    assert store.row(1) is None


def test_one_image_one_pair(store):
    store.set(0, layout_id="l001", states=ALL)
    store.set(1, layout_id="l001", states={**ALL, "pliers": "absent"}, pair_with_previous=True)
    with pytest.raises(ValueError, match="只能属于一对"):
        store.set(2, layout_id="l001", states={**ALL, "pliers": "absent", "hammer": "absent"}, pair_with_previous=True)


def test_unpair_clears_partner(store):
    store.set(0, layout_id="l001", states=ALL)
    store.set(1, layout_id="l001", states={**ALL, "pliers": "absent"}, pair_with_previous=True)
    store.set(1, layout_id="l001", states={**ALL, "pliers": "absent"}, pair_with_previous=False)
    assert store.row(0)["pair_group_id"] == "" and store.row(1)["pair_group_id"] == ""


def test_unknown_needs_reason_and_resume(store, tmp_path):
    with pytest.raises(ValueError, match="原因"):
        store.set(0, layout_id="l001", states={**ALL, "pliers": "unknown"})
    store.set(0, layout_id="l001", states=ALL)
    again = LabelStore(store.raw_dir, "s001", store.csv_path, data_dir=store.data_dir)
    assert again.row(0)["layout_id"] == "l001" and again.first_unlabeled() == 1
