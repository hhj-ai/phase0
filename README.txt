Phase0 环境：求稳版（CPU 下载 wheelhouse + GPU 离线安装）

你现在遇到的核心问题，其实是三个“坑”叠在一起：
1) pip 被环境/全局配置强制指向 pip.sankuai.com（HTTP），导致 numpy/pycocotools 等包查不到；
2) 你用 pip download 加了 --platform/--python-version 等 cross-platform 选项，但没加 --only-binary=:all:，pip 会直接报错（这是 pip 的硬性规则）；
3) GPU 节点里找不到 python3.10，所以 venv 只能用 3.8 创建，后续很多 wheel（cp310）根本装不上。

这套脚本的策略：
- cpu.sh：只负责“联网下载 wheelhouse”（目标平台：manylinux2014_x86_64 + cp310），并且强制用官方 PyPI（--isolated + -i https://pypi.org/simple）。
- gpu.sh：在 GPU 节点找 python3.10，用它创建 venv，再从 wheelhouse 离线安装。sdist-only 的包（qwen-vl-utils）会在 GPU 节点里 build wheel。

使用方式（建议流程）：
A) 在能联网的登录机/CPU 节点：
   bash cpu.sh
   看到 [P0][cpu] DONE 即 wheelhouse 准备好

B) 到 GPU 节点（必须有 python3.10）：
   # 你可以先 module load / conda activate 一个 py310
   bash gpu.sh
   source venv/p0_env/bin/activate

如果 GPU 节点确实没有 python3.10：
- 你只能先在有 python3.10 的地方创建好 venv 目录再 rsync 过去，或者让平台侧提供 python3.10（module/conda）。
