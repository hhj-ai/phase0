# Phase0 环境脚本（v3）

这套脚本把责任拆开：

- `cpu.sh`：只做一件事——把依赖下载成 **Python 3.10 (cp310)** 的 wheels，放到 `offline_wheels/py310/`。
  - 不创建 venv
  - 不修改你现有环境
  - 不动你的代码

- `gpu.sh`：在 GPU 节点创建/复用 `venv/p0_env`，并从 wheelhouse **离线安装**依赖。

- `run.sh`：跑一次完整 Phase0（probe -> worker -> analyze -> summary）。

## 1) 修复“文件被删”

你看到的大量 `deleting transformers/...` 是典型的 `rsync --delete` 或者错误的清理逻辑导致的。

如果你代码是 git 托管：

```bash
cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/phase0

git status
# 恢复工作区文件
git restore .
# 如果有未跟踪文件乱入再清理（会删掉未跟踪文件）
# git clean -fd
```

## 2) 重新下载 wheelhouse（有网的机器）

```bash
cd phase0
bash cpu.sh
```

默认优先走官方 PyPI；如果外网不通、公司源更快：

```bash
P0_INDEX_URL=https://pip.sankuai.com/simple \
P0_EXTRA_INDEX_URL=https://pypi.org/simple \
bash cpu.sh
```

## 3) GPU 节点离线安装

```bash
cd phase0
bash gpu.sh
```

## 4) 跑实验

```bash
bash run.sh
```

---

# 这份代码在验证什么（Phase0 的“科学假说”）

它在做一个“可证伪推断”的最小验证：

1. **同一张图，同一个问题**，给模型两种视觉输入：
   - 原图
   - 替换/扰动后的图（把关键证据区域拿走，或做某种替换）

2. 在若干层（vision / cross-attn 相关层）抽取表示，计算两次前向的差异（比如 JS/KL/某种距离汇总成 CED）。

3. 直觉：
   - 如果模型的答案真是“看图得出”，拿走证据后内部表示应该明显变化。
   - 如果模型在“胡说”（主要靠语言先验），拿走证据后表示变化会更小。

4. Phase0 的输出最终会做一个非常硬核的检验：
   - 用这个差异分数当作一个“hallucination detector”，看能不能把 **hallucination vs correct_positive** 分开（AUC）。

所以 Phase0 本质是在验证：**“证据敏感性”能不能成为 VLM 幻觉检测的信号**。
