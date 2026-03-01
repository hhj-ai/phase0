# phase0: 这段代码在验证什么（简版）

从你跑出来的日志看，phase0 主要在做一件事：

- 用 Qwen3-VL 这类 VLM（Vision-Language Model）跑一组输入；
- 把图像侧的 token（视觉 patch / visual tokens）在不同层做“捕获/替换/对比”；
- 观察输出分布（例如 logits/答案）是否随“视觉证据变化”而变化。

你看到的：
- **Expected image tokens: 1040**
- **Captured image tokens: 260**
- **ratio=4**

这个 ratio=4 往往对应 **2×2 的 patch merge（spatial_merge_size=2）**：视觉 token 数从 1040 缩到 260（除以 4）。
也就是：模型/预处理在“压缩视觉分辨率”，phase0 在确认这一点是不是按预期发生，并且后续替换实验是在同一个 token 对齐空间上做的。

（如果你愿意把 main.py / config 里“捕获/替换”的具体参数贴一段，我能把验证目标写成更精确的一句话。）
