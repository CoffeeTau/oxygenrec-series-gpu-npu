# v2 Full GPU固定参考运行记录

> 来源：用户于2026-09-12提供的GPU服务器运行结果。
> 性质：服务器运行结果摘录；仅保留可确认的配置、聚合值、哈希与验收边界。

## 执行

- 环境：GPU服务器，NVIDIA L20，`cuda:0`
- Git commit：`df26747ed289d7e9075717500817042e4bff6d67`
- 命令：`CUDA_VISIBLE_DEVICES=0 bash run_gpu_v2_migration_reference.sh`

## 终端聚合结果

```text
OK stage=v2_migration_static_inventory files=9 platform_specific_refs=11 official_tool_required=True output=checkpoints/device_alignment/v2_full/support_inventory/gpu/v2_migration_inventory.json
OK stage=v2_gpu_reference device=cuda:0 samples=8 loss=6.092291 logit_steps=6 greedy_shape=(8, 6) beam_shape=(8, 4, 6) checkpoint_sha256=02553b1ed7b8bbb5a1b1511134dbe06a60d7362923445908c5d6daca5c7ecabd artifact_sha256=f225f0a9731c3948fae1e3ed4d267aa77f0b199a76c6f890b75410e3393c4773 output=checkpoints/device_alignment/v2_full/gpu/v2_full_gpu_reference.pt
```

Transformer Encoder输出了`enable_nested_tensor`相关warning；它没有终止执行，最终
reference成功写出，因此不作为本轮失败项。

## Reference摘要

| 字段 | 值 |
|---|---|
| schema/protocol | `1` / `oxygenrec_v2_full_fixed_batch_fp32` |
| dtype | `float32` |
| source tracked dirty | `false` |
| source untracked count | `0` |
| SID registry version | `rq-control-l3-w256-s17-initkmeans++` |
| validation samples | `8` |
| sample seed | `17` |
| list size / SID levels | `2 / 3` |
| max history items | `20` |
| beam width | `4` |
| loss | `6.092290878295898` |
| logit shapes | `6 × [8,256]` |
| greedy shape | `[8,6]` |
| beam shape | `[8,4,6]` |
| gradient tensors | `80` |
| max parameter delta | `0.0003108978271484375` |

Performance字段的scope为`eval_forward_only`：warmup `3`步、计时`10`步，总耗时
`0.03563818900147453s`，均值`0.003563818900147453s/step`，吞吐
`2244.782976954581 samples/s`，peak allocated memory `16045568`字节。

## Missing gradients解释

摘要列出16个missing gradient：

- `history_context_adapter`、`history_context_query/key/value`共8个参数；
- `query_adapter`共4个参数；
- `item_adapter`共4个参数。

当前配置中`use_history_context_instruction=false`、`instruction_feature_size=0`、
`q2i_weight=0`，上述分支未参与本次Full预训练目标，因此没有梯度符合配置预期。
这不代表所有模型参数都已在本协议中覆盖；NPU比较必须要求missing列表与GPU完全相同。

## 完整性与结论边界

- checkpoint SHA-256：`02553b1ed7b8bbb5a1b1511134dbe06a60d7362923445908c5d6daca5c7ecabd`
- reference artifact SHA-256：`f225f0a9731c3948fae1e3ed4d267aa77f0b199a76c6f890b75410e3393c4773`
- 六组logits、greedy、beam、80组梯度和参数delta的具体tensor保存在reference `.pt`中，JSON只保存摘要。
- 本轮证明GPU参考生成链通过，不证明NPU已支持模型，也不证明两端精度一致。

## 关联

- 主日志：[`复现实验日志.md`](../../复现实验日志.md)
- 迁移计划：[`docs/npu_migration_plan.md`](../../../docs/npu_migration_plan.md)
