"""实验 yaml 只改 mix，共享超参与 contracts/training_mix.md 的预设一致。"""

from pathlib import Path

import pytest

from qwen_vlm.training.config import load_train_config

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs" / "train"

PRESETS = {
    "B1": {"include_unknown": True, "aux_presence": "none", "hard_negative_weight": 1.0, "balance_states": False},
    "M": {"include_unknown": True, "aux_presence": "pairs", "hard_negative_weight": 3.0, "balance_states": True},
    "A1": {"include_unknown": True, "aux_presence": "none", "hard_negative_weight": 3.0, "balance_states": True},
    "A2": {"include_unknown": True, "aux_presence": "pairs", "hard_negative_weight": 1.0, "balance_states": True},
    "A3": {"include_unknown": False, "aux_presence": "pairs", "hard_negative_weight": 3.0, "balance_states": True},
}


def test_presets_share_hyperparameters():
    base = load_train_config(CONFIGS / "base.yaml")
    assert base["lora"]["r"] == 16
    assert base["lora"]["alpha"] == 32
    assert base["lora"]["dropout"] == pytest.approx(0.05)
    assert base["optim"]["lr"] == pytest.approx(1e-4)
    assert base["optim"]["micro_batch_size"] * base["optim"]["grad_accum_steps"] == 16
    assert base["pixels"]["max_pixels"] == 512 * 512
    assert base["mix"]["aux_ratio"] == pytest.approx(1.0)
    for name, expected in PRESETS.items():
        cfg = load_train_config(CONFIGS / f"{name}.yaml")
        assert cfg["experiment_id"] == name
        for key, value in expected.items():
            assert cfg["mix"][key] == value
        assert cfg["lora"]["r"] == base["lora"]["r"]
        assert cfg["lora"]["alpha"] == base["lora"]["alpha"]
        assert cfg["lora"]["dropout"] == pytest.approx(base["lora"]["dropout"])
        assert cfg["optim"]["lr"] == pytest.approx(base["optim"]["lr"])
        assert cfg["optim"]["max_steps"] == base["optim"]["max_steps"]
        assert cfg["optim"]["grad_accum_steps"] == base["optim"]["grad_accum_steps"]
        assert cfg["pixels"]["max_pixels"] == base["pixels"]["max_pixels"]
        assert cfg["mix"]["aux_ratio"] == pytest.approx(base["mix"]["aux_ratio"])


def test_smoke_is_short_and_uses_same_pixel_budget():
    smoke = load_train_config(CONFIGS / "smoke.yaml")
    assert smoke["experiment_id"] == "train_smoke"
    assert smoke["optim"]["max_steps"] == 30
    assert smoke["optim"]["warmup_steps"] == 2
    assert smoke["optim"]["lr"] == pytest.approx(2e-4)
    assert smoke["pixels"]["max_pixels"] == 512 * 512
    assert smoke["lora"]["r"] == 16
    assert smoke["export_adapter_dir"] == "artifacts/adapters/smoke"
    assert smoke["data"]["train_jsonl"].endswith("data/smoke/synthetic/splits/v0/train.jsonl")
