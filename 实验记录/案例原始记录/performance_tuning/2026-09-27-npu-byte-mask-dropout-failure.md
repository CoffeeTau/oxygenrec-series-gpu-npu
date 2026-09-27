# NPU ByteMask Dropout A/B失败记录

> 记录类型：服务器运行结果摘录
> 实验编号：`E151`
> 服务器时间：`2026-09-27 09:54:26`
> 服务器产物根路径：`checkpoints/performance_optimization/npu_byte_mask_dropout_20260926/`
> 主日志：[`E151`](../../复现实验日志.md)
> 案例分析：[`v2 BF16单卡性能调优分析`](../../案例分析/v2_bf16_single_card_performance_tuning.md)

## 1. 已确认执行阶段

命令已通过control结果文件存在性检查并进入ByteMask treatment。treatment在第一个warmup前向中，
经过`OxygenRECModel.forward()`、`_behavior_instruction_prompt()`与
`behavior_instruction_adapter`内的Dropout时失败，未进入100步正式测量，未生成treatment JSON。

本次回传没有包含control JSON的聚合数值，因此不从终端局部截图补造性能结果。

## 2. 原始关键报错

```text
Warning: torch_npu.dropout_with_byte_mask is deprecated and will be removed in future version.

torch_npu.contrib.module.npu_modules.py, line 45, in forward
return F.dropout_with_byte_mask(input1, self.p, self.training, self.inplace)

RuntimeError: Current device only support aclnn operator, but current operator
dropout_with_byte_mask do not have aclnn implementation

ERR00007 PTA feature not supported
treatment result missing; stop before comparison
```

## 3. 事实边界

- 故障发生在ByteMask算子分派阶段，不是batch 4096容量不足、输入数据、checkpoint或loss异常；
- 警告和运行时错误共同表明该接口是旧实现，目标设备只允许ACLNN路径，而它没有ACLNN实现；
- `treatment result missing`是命令对缺失JSON的预期保护，不是独立故障；
- 标准Dropout在此前基线和Profile中实际走`aclnnDropoutV3`，仍是当前支持路径。

## 4. 决策

不尝试通过环境开关强制启用已弃用旧算子，不更换驱动/软件栈来迁就该候选，也不关闭dropout。
该路线以`[目标机不支持-未进入测量-未采纳]`关闭。下一轮直接转向训练循环中的逐step loss主机同步。
