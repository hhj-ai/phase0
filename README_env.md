# Phase0 环境脚本（求稳版 v5）

这套脚本的原则：**不碰你的代码、不清理 git 目录，只处理 venv / wheelhouse。**

- `cpu.sh`（联网机器跑）
  - 创建一个很小的 downloader venv（`venv/p0_cpu_downloader`），只用来跑 `pip download`。
  - 强制从 **官方 PyPI** 下载，并固定目标为 **manylinux2014_x86_64 + Python 3.10 (cp310)**。
  - 产物是 `offline_wheels/py310/*.whl`（wheelhouse）。

- `gpu.sh`（GPU/离线机器跑）
  - 创建/复用运行 venv（默认 `venv/p0_env`，`--system-site-packages`）。
  - 只从 wheelhouse 离线安装（`--no-index --find-links`）。
  - 做一次 import 自检（numpy/torch/transformers/datasets/pycocotools）。

- `run.sh`
  - probe -> worker -> analyze -> summary 跑一遍 Phase0。

> 你可以用 `sh cpu.sh` / `sh gpu.sh` 跑：脚本会自动 `exec bash`，避免 dash/sh 的兼容坑。

---

## 1) 如果你刚刚“文件被删”了（git 托管的情况）

在 phase0 代码目录：

```bash
cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/phase0

git status
# 恢复被改/被删的受控文件
git restore .

# 如果你确认需要把未跟踪文件也清掉（会删！慎用）
# git clean -fd
```

> 这套 v5 脚本本身**不会**对你的 repo 做 `git clean` / `rm -rf` 之类的“扫地机器人行为”。

---

## 2) 在有网机器重新下载 wheelhouse

```bash
cd phase0
bash cpu.sh
```

脚本用 `--isolated` 忽略你机器上的 pip 配置（避免被 `http://pip.sankuai.com/simple` 之类的配置劫持）。

---

## 3) 在 GPU 节点离线安装

```bash
cd phase0
bash gpu.sh
```

如果它报 “找不到 python>=3.9”，说明你当前 shell 里的 python 太旧（例如 3.8）。
解决方式就是**先切到带 python3.10 的环境**再跑（conda/module），或者显式：

```bash
P0_PYTHON_RUN=/path/to/python3.10 bash gpu.sh
```

---

## 4) 为什么脚本不会“自动把你终端切进 (p0_env)？”

因为 `sh/b​​ash xxx.sh` 是开一个子进程跑脚本；脚本里的 `source venv/bin/activate` 只影响它自己，
不可能反向修改你外面那个终端的环境变量。

如果你想让**当前终端**也进入 venv，手动执行：

```bash
source venv/p0_env/bin/activate
```

---

## 5) 这份代码在验证什么（Phase0 在做的事）

Phase0 是一个“先把地基钉稳”的验证：它在检查 **VLM 的视觉证据到底有没有进到推理里**。

核心思想是做“证据敏感性”测试：

1. 对同一张图同一个问题，跑两次前向：
   - 原图
   - 做过替换/扰动的图（把关键证据区域拿走，或用某种替换方式干预）
2. 在中间表征（视觉 token / 融合层相关表征）上，计算两次前向的差异（代码里你看到的 JS_sum / CED 等汇总指标）。
3. 直觉上：
   - 如果答案真依赖视觉证据，拿走证据后表征应该明显变化。
   - 如果答案主要靠语言先验在“自信胡说”，拿走证据后变化会更小。

所以 Phase0 的目标不是“刷分”，而是在验证一个可检验假说：

> **表征差异（证据敏感性）能不能作为 VLM 幻觉/正确的区分信号。**
