# v2 Full GPU/NPU FP32短训练对照

日期：2026-09-15

信息来源：GPU与NPU服务器运行结果

## 1. 判定

`[通过-v2 Full FP32迁移短训练]`

两端使用同一代码提交和相同训练输入，均完成20步FP32续训、有限梯度、checkpoint保存恢复
以及恢复后前向。本记录只确认模型迁移短训练链可用，不代表最终推荐精度、正式性能或多卡验收。

## 2. 可比性前提

| 项目 | GPU | NPU | 判断 |
|---|---|---|---|
| Git commit | `9633639329cc569c18b458683980ecef38d0c9f5` | 相同 | 通过 |
| 工作树 | tracked dirty=false，untracked=0 | 相同 | 通过 |
| events SHA-256 | `3745aa83238b1e6d44d8fda209807899f420084398f94ddf745f3cbcfecbf9e7` | 相同 | 通过 |
| 起始checkpoint SHA-256 | `02553b1ed7b8bbb5a1b1511134dbe06a60d7362923445908c5d6daca5c7ecabd` | 相同 | 通过 |
| SID registry SHA-256 | `5f12417ff5e60818753be8ec2d8c5dece350068a30a36d25f17bffab8dead29a` | 相同 | 通过 |
| registry版本 | `rq-control-l3-w256-s17-initkmeans++` | 相同 | 通过 |
| 训练配置 | seed=17，steps=20，batch=64，样本=5000，lr=0.0003，AdamW | 相同 | 通过 |

## 3. 运行环境

| 项目 | GPU服务器 | NPU服务器 |
|---|---|---|
| 设备 | NVIDIA L20，`cuda:0` | Ascend950DT_9572，`npu:0` |
| Python | `3.12.3` | `3.11.6` |
| PyTorch | `2.11.0a0+a6c236b9fd.nv26.03.46836102` | `2.10.0+cpu` |
| TorchNPU | 不适用 | `2.10.0.post5.dev20260910` |
| 精度 | FP32，无autocast | FP32，无autocast |

NPU的TorchNPU版本不同于旧环境快照记录的post4开发版，因此最终迁移验收前仍需重新采集
完整NPU环境信息。本次训练摘要已经固定当前实际参与计算的Python、PyTorch和TorchNPU版本。

## 4. 训练与保存恢复

| 指标 | GPU | NPU | 差异或判断 |
|---|---:|---:|---:|
| first loss | 5.600089 | 5.597483 | 绝对差0.002605 |
| last loss | 5.788480 | 5.790234 | 绝对差0.001754 |
| 20步平均loss | 5.707635 | 5.707190 | NPU-GPU=-0.000445 |
| 逐步loss平均绝对差 | — | — | 0.008988，即GPU平均loss的约0.1575% |
| 逐步loss最大绝对差 | — | — | 0.023938，发生在第19步 |
| loss差RMSE | — | — | 0.010912 |
| gradient L2 first | 0.810402 | 0.807149 | NPU低约0.40% |
| gradient L2 last | 0.931534 | 0.925463 | NPU低约0.65% |
| gradient max abs | 0.040667 | 0.041919 | NPU高约3.08% |
| gradient tensors | 80 | 80 | 一致 |
| all losses finite | true | true | 通过 |
| reload state match | true | true | 通过 |
| reload forward finite | true | true | 通过 |
| reload forward loss | 5.502078 | 5.495470 | 绝对差0.006608 |
| 最终状态 | passed | passed | 通过 |

不同mini-batch上的20步loss本来就会波动，不能用“末步必须低于首步”作为本测试的通过条件。
GPU与NPU还使用各自的随机数实现和底层算子，因此独立训练产生不同的输出checkpoint哈希是
预期现象；这里要求相同的是起始代码、数据、checkpoint和registry。

## 5. 非正式性能观测

| 项目 | GPU | NPU |
|---|---:|---:|
| elapsed seconds | 1.1950 | 1.6694 |
| mean step seconds | 0.05975 | 0.08347 |
| steps per second | 16.7363 | 11.9805 |
| peak allocated memory | 80.22 MiB | 104.87 MiB |

这些数值来自20步短训练，未独立隔离预热/编译，且梯度统计会触发设备同步；不同后端的
allocated memory语义也不能直接等同。因此只保存原始观测，不形成GPU/NPU性能优劣结论。

## 6. NPU环境启动前提

NPU首次运行曾报告TorchNPU后端扩展加载失败。补充动态库路径并依次加载driver和CANN环境后，
同一训练命令成功：

```bash
export LD_LIBRARY_PATH=/usr/lib64:${LD_LIBRARY_PATH:-}
source /usr/local/Ascend/driver/bin/setenv.bash
source /usr/local/Ascend/cann/set_env.sh
```

该问题归类为目标服务器环境初始化问题，不归类为模型训练失败。

## 7. 下一步

1. GPU与NPU分别使用统一入口执行BF16短训练；
2. 检查两端autocast确实开启、dtype为BF16、loss/梯度有限且checkpoint保存恢复通过；
3. BF16通过后再设计固定验证集指标比较；
4. 仅在验证指标或数值出现异常时启用大JSON探针或官方精度调试工具；
5. 正式性能测试前确认旧eval路径的Transformer融合算子CPU fallback状态，并刷新NPU环境快照。
