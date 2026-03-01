# phase0 (conda-stable bootstrap)

这版的目标很简单：**求稳**。  
核心策略：用 conda（优先系统 conda；没有就自动装本地 Miniforge）创建 *python=3.10* 的前缀环境，然后装依赖。

## 一键装 CPU 环境
```bash
bash cpu.sh
```

验证：
```bash
bash tools/p0_run.sh cpu python -c "import pycocotools; print('pycocotools OK')"
```

## 一键装 GPU 环境
```bash
bash gpu.sh
```

验证：
```bash
bash tools/p0_run.sh gpu python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

## 不重复下载/安装
- `cpu.sh` / `gpu.sh` 会对 `requirements.*.txt` 做 sha256。
- requirements 未变且 stamp 存在时，会跳过安装。
- 需要强制重装：`P0_FORCE=1 bash cpu.sh` / `P0_FORCE=1 bash gpu.sh`

## 为什么脚本不会“自动切换”你当前 shell 的环境？
`bash cpu.sh` 是在一个子进程里运行，子进程无法改变父 shell 的激活状态。  
所以这里用 `conda run -p <env>` 来执行命令，不需要你手动 activate。

> 你仍然可以手动激活：  
> `source .runtime/miniforge3/etc/profile.d/conda.sh && conda activate /path/to/venv/p0_gpu`
