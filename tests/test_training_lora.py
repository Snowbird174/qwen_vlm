"""LoRA 正则只匹配语言模型投影，不匹配视觉分支里的同名模块。"""

from qwen_vlm.training.lora import LANGUAGE_LORA_TARGET_PATTERN, module_matches_lora_target

_LANGUAGE_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def test_pattern_matches_language_projections_only():
    names = [
        "model.language_model.layers.0.self_attn.q_proj",
        "model.language_model.layers.0.self_attn.k_proj",
        "model.language_model.layers.0.self_attn.v_proj",
        "model.language_model.layers.0.self_attn.o_proj",
        "model.language_model.layers.27.mlp.gate_proj",
        "model.language_model.layers.27.mlp.up_proj",
        "model.language_model.layers.27.mlp.down_proj",
        "model.language_model.layers.0.self_attn.q_norm",
        "model.language_model.embed_tokens",
        "lm_head",
        "model.visual.blocks.0.attn.qkv",
        "model.visual.blocks.0.attn.proj",
        "model.visual.blocks.0.attn.q_proj",
        "model.visual.blocks.0.attn.k_proj",
        "model.visual.blocks.0.mlp.gate_proj",
        "model.visual.blocks.0.mlp.down_proj",
        "model.visual.merger.linear_fc1",
        "model.visual.merger.linear_fc2",
        "model.visual.deepstack_merger_list.0.linear_fc1",
        "model.visual.patch_embed.proj",
    ]
    matched = [name for name in names if module_matches_lora_target(name)]
    assert matched == [
        "model.language_model.layers.0.self_attn.q_proj",
        "model.language_model.layers.0.self_attn.k_proj",
        "model.language_model.layers.0.self_attn.v_proj",
        "model.language_model.layers.0.self_attn.o_proj",
        "model.language_model.layers.27.mlp.gate_proj",
        "model.language_model.layers.27.mlp.up_proj",
        "model.language_model.layers.27.mlp.down_proj",
    ]
    assert all("visual" not in name for name in matched)
    assert "language_model" in LANGUAGE_LORA_TARGET_PATTERN


def test_all_language_layers_match_and_visual_twins_do_not():
    matched = []
    for layer in range(28):
        for suffix in _LANGUAGE_SUFFIXES:
            name = f"model.language_model.layers.{layer}.{suffix}"
            assert module_matches_lora_target(name)
            matched.append(name)
    assert len(matched) == 28 * 7
    for block in range(24):
        assert not module_matches_lora_target(f"model.visual.blocks.{block}.attn.q_proj")
        assert not module_matches_lora_target(f"model.visual.blocks.{block}.attn.k_proj")
        assert not module_matches_lora_target(f"model.visual.blocks.{block}.attn.v_proj")
        assert not module_matches_lora_target(f"model.visual.blocks.{block}.attn.o_proj")
        assert not module_matches_lora_target(f"model.visual.blocks.{block}.mlp.gate_proj")
        assert not module_matches_lora_target(f"model.visual.blocks.{block}.mlp.up_proj")
        assert not module_matches_lora_target(f"model.visual.blocks.{block}.mlp.down_proj")
        assert not module_matches_lora_target(f"model.visual.blocks.{block}.attn.qkv")
        assert not module_matches_lora_target(f"model.visual.merger.linear_fc{block}")
