# NPU融合AdamW首轮A/B失败记录

> 记录类型：服务器运行结果摘录  
> 实验编号：`E145`  
> 服务器日期：`2026-09-24`  
> 服务器：`server-118`  
> 服务器产物路径：`checkpoints/performance_optimization/npu_fused_adamw_20260924/`

## 1. Control结果

```text
repeat=1/3 samples_per_second=26819.998 step_ms=152.722
repeat=2/3 samples_per_second=27614.505 step_ms=148.328
repeat=3/3 samples_per_second=27372.037 step_ms=149.642
```

- 吞吐均值：`27268.8467 samples/s`；
- 吞吐中位数：`27372.037 samples/s`；
- 吞吐CV：约`1.2192%`；
- control JSON成功生成，应继续保留为本次失败实验的原始证据。

## 2. Treatment失败

失败发生在第一个warmup step的梯度清零：

```text
optimizer.zero_grad(set_to_none=True)
ValueError: set_to_none is not supported in fused optimizers
```

该错误发生在`NpuFusedAdamW`构造和checkpoint optimizer state加载之后、第一次前向之前。因此当前
证据排除了模型算子、loss、NPU硬件和checkpoint加载作为本次直接根因。

## 3. 连锁错误

treatment未生成`npu_bf16_performance.json`，比较脚本随后读取该文件并出现`FileNotFoundError`。
这是首个错误未中止后续命令造成的连锁错误，不是第二个独立实验故障。

## 4. 处理决策

- 不删除本轮目录；
- 不只修改treatment，否则A/B会同时改变优化器和zero-grad语义；
- 新一轮control和treatment统一使用`zero_grad(set_to_none=False)`；
- JSON显式记录`zero_grad_mode`，比较脚本拒绝两组模式不一致；
- 缺失结果文件时，比较脚本返回简短错误并停止。

118与119机器按用户确认几乎完全一致，可用于历史横向参考；融合优化器收益仍按同机成对A/B判断。
