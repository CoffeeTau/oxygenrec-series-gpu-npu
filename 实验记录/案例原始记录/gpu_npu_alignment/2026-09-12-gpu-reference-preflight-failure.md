# GPU reference前置检查失败（截图转录版）

> 来源：用户于2026-09-12提供的GPU服务器终端截图。
> 性质：截图转录，不是服务器原始日志文件；只记录截图中可辨认且与判断直接相关的字段。

## 执行环境与命令

- 环境：GPU服务器
- Git HEAD：`19a76bd`
- 命令：`CUDA_VISIBLE_DEVICES=0 bash run_gpu_v2_migration_reference.sh`

## 可辨认输出

```text
OK stage=v2_migration_static_inventory files=9 platform_specific_refs=11
official_tool_required=True
output=checkpoints/device_alignment/v2_full/support_inventory/gpu/v2_migration_inventory.json

RuntimeError: GPU reference export requires a clean Git worktree with a known commit
```

随后执行`git status --short --untracked-files=all`，截图中可见条目全部以`??`开头，
包括：

- `data/sft/`下的reasoning candidate/audit/review JSON或JSONL；
- `models/Qwen3-4B-Instruct-2507/`下的模型配置、tokenizer与safetensors权重；
- `outputs/review/`下的LLM、Qwen、EA-TOSD与checkpoint comparison产物。

截图没有显示`M`、`A`或`D`形式的已跟踪文件变化。

## 证据边界

- 静态清单阶段已成功；
- Transformer的nested tensor信息是warning，不是本次终止原因；
- 本截图不能证明reference数值阶段完成，也不能证明GPU/NPU精度对齐；
- 根因是Git洁净检查把未跟踪实验资产纳入`dirty`，不代表模型或CUDA执行失败。

## 关联

- 主日志：[`复现实验日志.md`](../../复现实验日志.md)
- 迁移计划：[`docs/npu_migration_plan.md`](../../../docs/npu_migration_plan.md)
