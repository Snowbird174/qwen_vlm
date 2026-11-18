"""网页标注工具（Gradio）。逻辑在 label_store.py，这里只有界面。"""

from __future__ import annotations

import gradio as gr

from qwen_vlm.common.categories import CATEGORY_IDS, NAME_ZH
from qwen_vlm.data.label_store import LabelStore

STATE_CHOICES = [("在", "present"), ("不在", "absent"), ("看不清", "unknown")]
CONDITION_CHOICES = [
    ("手或纸遮挡", "occlusion"),
    ("手入画但没挡住", "hand_no_occlusion"),
    ("失焦/模糊", "defocus"),
    ("白板没拍全", "partial_frame"),
    ("换了位置", "position_shift"),
    ("光线偏暗", "lighting_dim"),
    ("反光", "reflection"),
]
HARD_CHOICES = [("表笔线 vs 线缆", "probe_leads_vs_cable"), ("钳子 vs 其他金属工具", "pliers_vs_metal_tool")]

HELP = """**判断规则**：看得见才选「在」；白板整块都拍到了、确实没有才选「不在」；被挡住或看不清选「看不清」，不要猜。
表笔线算万用表的一部分，「线缆」只指单独的一束线。
**摆放编号**：同一次摆放下拍的几张（只拿走/放回/挪动东西）用同一个编号；重新摆一桌就点「新摆放」。
**与上一张配对**：这张和上一张是同一摆放、只拿走或放入了**一件**时勾选。"""


def build_app(store: LabelStore) -> gr.Blocks:
    def view(i: int):
        i = max(0, min(int(i), len(store) - 1))
        row = store.row(i)
        prev = store.previous_labeled(i)
        if row is not None:
            states = [row[c] for c in CATEGORY_IDS]
            layout = row["layout_id"]
            reason = row["unknown_reason"].split("=", 1)[1] if "=" in row["unknown_reason"] else ""
            tags = [t for t in row["condition_tags"].split("|") if t]
            hard = [t for t in row["hard_negative_tags"].split("|") if t]
            paired = bool(row["pair_group_id"]) and prev is not None and prev["pair_group_id"] == row["pair_group_id"]
            hint = "已标注，可修改后重新保存。"
        else:
            states = [prev[c] for c in CATEGORY_IDS] if prev else [None] * 5
            layout = prev["layout_id"] if prev else "l001"
            reason, tags, hard, paired = "", [], [], False
            hint = "未标注：状态和摆放编号**沿用了上一张，请逐项核对**。" if prev else "未标注。"
        header = (
            f"### 第 {i + 1} / {len(store)} 张　（{store.status(i)}）　已完成 {store.n_done()} / {len(store)}\n"
            f"`{store.rel_paths[i]}`　{hint}"
        )
        return [i, str(store.images[i]), header, layout, *states, reason, paired, tags, hard, ""]

    def save(i, layout, *rest):
        states = dict(zip(CATEGORY_IDS, rest[:5]))
        reason, paired, tags, hard = rest[5:9]
        try:
            msgs = store.set(
                int(i), layout_id=layout, states=states, unknown_reason=reason,
                condition_tags=tags, hard_negative_tags=hard, pair_with_previous=paired,
            )
        except ValueError as exc:
            out = view(i)
            out[-1] = f"⚠ 没有保存：{exc}"
            return out
        out = view(int(i) + 1 if int(i) + 1 < len(store) else int(i))
        out[-1] = "已保存第 %d 张。%s" % (int(i) + 1, "；".join(msgs))
        return out

    def skip(i):
        store.skip(int(i))
        out = view(int(i) + 1 if int(i) + 1 < len(store) else int(i))
        out[-1] = f"第 {int(i) + 1} 张已标为废片（不进入数据集）。"
        return out

    def check():
        errors, n = store.check()
        if not errors:
            return f"✅ 检查通过：{len(store.rows)} 张照片已标注，可展开成 {n} 条样本。标注表：`{store.csv_path}`"
        shown = "\n".join(f"- {e}" for e in errors[:30])
        more = f"\n- ……共 {len(errors)} 条" if len(errors) > 30 else ""
        return f"❌ 发现 {len(errors)} 个问题：\n{shown}{more}"

    with gr.Blocks(title="器材点检标注") as demo:
        idx = gr.State(store.first_unlabeled())
        header = gr.Markdown()
        with gr.Row():
            with gr.Column(scale=3):
                image = gr.Image(type="filepath", interactive=False, height=600, label="照片")
            with gr.Column(scale=2):
                with gr.Row():
                    layout = gr.Textbox(label="摆放编号", scale=3)
                    new_layout = gr.Button("新摆放", scale=1)
                radios = [gr.Radio(STATE_CHOICES, label=NAME_ZH[c]) for c in CATEGORY_IDS]
                reason = gr.Textbox(label="看不清的原因（有「看不清」时必填）", placeholder="例如：被手挡住")
                paired = gr.Checkbox(label="与上一张配对（同一摆放，只拿走或放入了一件）")
                tags = gr.CheckboxGroup(CONDITION_CHOICES, label="拍摄情况（可多选，可不选）")
                hard = gr.CheckboxGroup(HARD_CHOICES, label="容易混淆的情况（可不选）")
                with gr.Row():
                    prev_btn = gr.Button("◀ 上一张")
                    save_btn = gr.Button("保存并下一张 ▶", variant="primary")
                with gr.Row():
                    skip_btn = gr.Button("这张是废片，跳过")
                    jump = gr.Number(label="跳到第几张", precision=0, minimum=1, maximum=len(store))
                    jump_btn = gr.Button("跳转")
                msg = gr.Markdown()
        with gr.Row():
            check_btn = gr.Button("检查全部标注")
        check_out = gr.Markdown()
        gr.Markdown(HELP)

        outputs = [idx, image, header, layout, *radios, reason, paired, tags, hard, msg]
        demo.load(view, inputs=idx, outputs=outputs)
        save_btn.click(save, inputs=[idx, layout, *radios, reason, paired, tags, hard], outputs=outputs)
        prev_btn.click(lambda i: view(int(i) - 1), inputs=idx, outputs=outputs)
        skip_btn.click(skip, inputs=idx, outputs=outputs)
        jump_btn.click(lambda n: view(int(n or 1) - 1), inputs=jump, outputs=outputs)
        new_layout.click(lambda: (store.next_layout_id(), False), outputs=[layout, paired])
        check_btn.click(check, outputs=check_out)
    return demo
