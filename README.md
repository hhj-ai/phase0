# Phase0 环境脚本（求稳版 v6）

## 你现在遇到的两个“看起来很玄学”的点
1) **cpu.sh “下载了”但你看不到 venv/p0_env/bin/activate**
- 之前的脚本在“下载 wheels”阶段就因为 pip 参数报错退出了（你贴的那条：需要 `--only-binary` 或 `--no-deps`），所以并没有走到“创建运行 venv”的步骤，自然也不会有 `venv/p0_env`。
- 另外，你命令行前缀的 `(p0_env)` 很可能是 **conda 环境名**，不等于我们要创建的 `phase0/venv/p0_env`。

2) **为什么脚本不能“自动切换到新环境”**
- 你用 `bash gpu.sh` / `sh cpu.sh` 跑的是**子进程**；子进程没法永久改你当前 shell 的 PATH / 变量。
- 所以最终还是要你在当前 shell 手动 `source venv/p0_env/bin/activate`。

## 这份脚本做了什么（核心稳健性）
- 总是用 **官方 PyPI（https://pypi.org/simple）** 下载，不吃你机器上的 pip 配置。
- wheelhouse 有 manifest（requirements/constraints 的 sha256），**一致就不重复下载**。
- 即使你当前只有 python3.8，也能先把 **cp310 的 wheels** 下载好（下载器 venv 用 3.8 也行）。
- 运行 venv 必须用 python>=3.10：会自动尝试
  - PATH 里的 python3.10 / python3
  - conda base（如果存在且是 3.10+）
  - 以及在 `ROOT_DIR/..` 下搜索已有的 `*/bin/python3.10`（最多 6 层）

## 用法
```bash
cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/phase0

bash cpu.sh
# or GPU
bash gpu.sh

# 进入环境（让当前 shell 生效）
source venv/p0_env/bin/activate
```

强制重下：
```bash
P0_FORCE_DOWNLOAD=1 bash cpu.sh
```

如果你知道 python3.10 的绝对路径（最稳）：
```bash
P0_PYTHON_RUN=/path/to/python3.10 bash cpu.sh
```

## 这段 Phase0 代码在验证什么（实验含义）
你这套 Phase0 更像是“可证伪视觉推断（FVI）”的一个 **pilot sanity check**：
- 目标不是跑 SOTA，而是验证：**当视觉证据被干预（替换/抹掉/打乱）时，模型的答案是否会按理改变**。
- 如果模型在“视觉证据被破坏”后依然给出高度自信且不变的回答，就说明它很可能在“凭语言先验瞎编”（confirmation bias）。
- 因此 Phase0 主要在做两件事：  
  1) 把图像输入转成视觉 token/特征，做不同类型的干预（mask / replace / noise 等）；  
  2) 观察生成答案的稳定性、logits/hidden states 的漂移，作为“是否真的依赖视觉”的证据。

（这就是你后面 Phase1/Phase2 可以扩展成系统性评测和训练目标的那个方向。）
