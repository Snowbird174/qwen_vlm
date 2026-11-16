# Git 进度管理约定

远程仓库：`https://github.com/zy-StarXH/qwen_vlm.git`（`origin`），主分支 `main`。

## 什么进仓库，什么不进

进仓库：源码、配置、接口、文档、测试、合成冒烟数据、小型运行记录（`run_manifest.json`、`metrics.json`、`predictions.jsonl`、日志摘要）、模型 manifest 与 SHA256。

不进仓库（见 `.gitignore`）：conda 环境 `envs/`、缓存 `cache/`、临时目录 `work/`、llama.cpp 源码与构建 `vendor/`、模型权重（safetensors、GGUF、adapter、检查点）、真实照片 `data/raw/`、交付包 `release/`。这些靠 manifest 里的 revision、commit 和 SHA256 复现；需要分享时另行打包（GitHub Release 或网盘）。

提交前确认 `git status` 里没有大文件：`git add -n . | head`，单文件超过 10MB 就停下来检查。

## 分支

- `main`：始终保持 `pytest` 全部通过、README 中的命令可以执行。
- 较大的工作开短分支，完成后合并回 `main`：
  - `data/<版本>`：真实数据接入与划分，例如 `data/real-v1`
  - `exp/<实验ID>`：某个训练实验的配置调整，例如 `exp/M-seed1`
  - `deploy/<内容>`：板端适配，例如 `deploy/jetson-bench`
  - `report/<章节>`：报告写作
- 一个人开发时，小改动直接提交到 `main` 也可以，但每次提交前都要跑 `pytest`。

## 提交

- 一个提交只做一件事。提交说明用中文或英文，写清楚"做了什么、为什么"，例如：`接入第一批真实照片（session01-04），更新 split v1`。
- 实验结果和产生它的代码放在同一个提交里，或者紧跟着提交，这样 `run_manifest.json` 里记录的 commit 能对上代码。
- 训练脚本会把当前 commit 写进 manifest；工作区有未提交改动时会标记 `dirty`。正式实验前先提交。

## 里程碑标签

用带说明的标签（`git tag -a <名字> -m "<说明>"`）标记阶段性状态，推送时带上 `--tags`：

| 标签 | 含义 | 状态 |
| --- | --- | --- |
| `cloud-ready` | 服务器侧环境、接口、训练和部署链路用合成数据跑通 | 已打 |
| `data-v1` | 第一批真实数据标注、校验、划分完成，测试集冻结 | 待做 |
| `baseline-b0` | B0 在真实测试集上的结果 | 待做 |
| `jetson-baseline` | 板端原模型跑通，日志已传回 | 待做 |
| `exp-main` | B1 / M / 消融完成 | 待做 |
| `opt-done` | 量化与分辨率的板端实测完成 | 待做 |
| `submission` | 课程提交版本 | 待做 |

`STATUS.md` 与标签同步更新。

## 常用命令

```bash
cd /root/gpufree-data/qwen_vlm && source env/activate.sh
envs/qwen_vlm/bin/python -m pytest -q tests     # 提交前
git status && git add -A && git commit -m "说明"
git push origin main                            # 需要 GitHub 用户名 + Personal Access Token
git tag -a data-v1 -m "第一批真实数据划分完成" && git push origin --tags
```

服务器访问 GitHub 要走本地代理（`127.0.0.1:7897`）。代理没开时 push 会报 `Failed to connect to 127.0.0.1 port 7897`。
