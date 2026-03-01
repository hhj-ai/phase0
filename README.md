# phase0 · 便携式（不依赖系统 Python）环境结构

这一套脚本的目标：**不管你当前节点的 `python` 是 3.8 还是没有 `python3.10` 命令，都能在 phase0 目录里自举出 Python 3.10 + venv，然后装依赖并跑 phase0 的 probe。**

## 你现在这套代码在“验证什么”（phase0 的意义）
从你贴出来的日志特征来看（`Expected 1040, captured 260, ratio=4.00` 这类输出），phase0 在做的是：
- **验证视觉 token/patch 的计数逻辑是否一致**（常见原因：Qwen/Qwen-VL 系列会做 patch merge / spatial_merge，把 4 个 patch 合成 1 个 token，所以期望 token 数会出现 4 倍关系）。
- **验证 hook/capture 的层输出是否对齐预期**（你抓到的视觉特征维度、token 数、层选择是否正确）。
- **做一个最小闭环的“环境+依赖+推理链路”自检**：能 import、能跑、能拿到稳定的统计指标（比如 JS/CED 之类的分布距离）。

> 具体是哪一段代码打印这些值：请直接 `grep -R "Expected" -n .` 或在 `main.py / p0_*.py` 里搜关键词定位。

## 使用方式（最稳、最少踩坑）
1) **在任意有网络的节点**（CPU/GPU 都行，但 CPU 更省）执行：
```bash
cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/phase0
bash cpu.sh
```

2) **在 GPU 节点**（有 CUDA）执行：
```bash
bash gpu.sh
```

3) 之后你可以用两种方式跑代码：

- 不激活环境，直接用 venv 里的 python：
```bash
./venv/p0_env/bin/python -V
./venv/p0_env/bin/python main.py  # 你自己的参数
```

- 或者激活（可选）：
```bash
source ./venv/p0_env/bin/activate
python -V
python main.py
```

> 说明：脚本**无法**“永久改变你当前 shell 的环境变量/激活状态”（子进程改不了父进程），所以我给了“直接用 venv/python 跑”的方式，最稳。

## 目录结构（脚本会自动生成）
- `.runtime/python310/`  自举的 CPython 3.10（不依赖系统 python）
- `.runtime/downloader_venv/` 仅用于下载/构建 wheel 的临时 venv
- `offline_wheels/py310/` 轮子仓（wheelhouse），成功后会写入 `.wheelhouse.ok`
- `venv/p0_env/` 你的运行 venv

## 常见坑的对应（这套脚本针对你对话里出现的问题）
- `pip.sankuai.com` 这种 http 源导致 “No matching distribution found”：脚本强制 `PIP_CONFIG_FILE=/dev/null` + `https://pypi.org/simple`。
- `pip download --platform/--python-version` 报 “必须 --only-binary=:all: 或 --no-deps”：脚本改为**用自举的 python3.10 直接 download 当前平台的 wheel/sdist**，不走 cross-plat 约束。
- `pycocotools` / `qwen-vl-utils` 只有 sdist：脚本会下载 sdist 并在本机 build wheel 放进 wheelhouse。
- 不重复下载：wheelhouse 做了指纹，requirements/constraints 没变就跳过。
