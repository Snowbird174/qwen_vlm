# 服务器部署冒烟

这些数字只说明服务器上的流程是否走通，不是板端性能，也不是精度结论。

## 像素档

三档 llama.cpp 日志里的 nx*ny 与 Transformers 的视觉 token 数一致。

| 档 | Transformers 视觉 token | llama.cpp nx*ny | prompt tokens | 服务是否截断上限 |
| --- | --- | --- | --- | --- |
| 384 | 130 | 130 | 385.0 | False |
| 512 | 234 | 234 | 489.0 | False |
| 768 | 540 | 540 | 795.0 | False |

prompt token 差值：{"prompt_tokens": [385.0, 489.0, 795.0], "delta_512_minus_384": 104.0, "delta_768_minus_512": 306.0, "expected_delta_512_minus_384": 104, "expected_delta_768_minus_512": 306}

## 与 Transformers bf16 的文本对照（像素档 512）

| 图 | 量化 | 原文一致 | 解析一致 | llama.cpp 解析 | finish_reason | 含中文 | item_states 次数 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| data/smoke/b0/bench_empty.png | f16 | True | True | ok | stop | False | 1 |
| data/smoke/b0/bench_hammer_screwdriver.png | f16 | False | False | ok | stop | False | 1 |
| data/smoke/b0/bench_multimeter.png | f16 | True | True | ok | stop | False | 1 |
| data/smoke/b0/bench_empty.png | q8_0 | True | True | ok | stop | False | 1 |
| data/smoke/b0/bench_hammer_screwdriver.png | q8_0 | True | True | ok | stop | False | 1 |
| data/smoke/b0/bench_multimeter.png | q8_0 | True | True | ok | stop | False | 1 |
| data/smoke/b0/bench_empty.png | q4_k_m | True | True | ok | stop | False | 1 |
| data/smoke/b0/bench_hammer_screwdriver.png | q4_k_m | False | False | ok | stop | False | 1 |
| data/smoke/b0/bench_multimeter.png | q4_k_m | False | False | ok | stop | False | 1 |

颜色和方向：runner 收到的是基类 `load_image` 之后的 RGB，再编码成 PNG 发给 llama-server。服务端用图像库按 PNG 解成 RGB，runner 里没有 OpenCV，也不会把通道换成 BGR。EXIF 方向由基类处理，有单测覆盖。中文问题走 UTF-8 JSON；模型若遵守提示词，正文是英文枚举的 JSON，表里的「含中文」因此可能是 false。
finish_reason 为 stop 表示生成停在 EOS 或停止符；length 表示撞上 max_new_tokens。`item_states` 次数大于 1 视为重复生成。

## 演示界面

图片模式调用：通过。服务器没有可用的采集摄像头，摄像头预览没有测。

