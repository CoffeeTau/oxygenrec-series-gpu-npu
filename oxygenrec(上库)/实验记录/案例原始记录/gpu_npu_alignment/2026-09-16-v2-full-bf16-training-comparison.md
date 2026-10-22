# v2 Full GPU/NPU BF16短训练对照

日期：2026-09-16

信息来源：GPU与NPU服务器运行结果

## 1. 判定

`[通过-v2 Full BF16迁移短训练]`

两端使用同一代码提交、相同训练输入和相同超参数，均完成20步BF16 autocast续训、有限梯度、
checkpoint保存恢复及恢复后前向。本记录不等价于所有算子均以BF16执行，也不是最终固定验证集
精度或正式性能结论。

## 2. 可比性与BF16启用状态

| 项目 | GPU | NPU | 判断 |
|---|---|---|---|
| Git commit | `d519176a522ded7517c1675e8260098df4486214` | 相同 | 通过 |
| 工作树 | tracked dirty=false，untracked=0 | 相同 | 通过 |
| events SHA-256 | `3745aa83238b1e6d44d8fda209807899f420084398f94ddf745f3cbcfecbf9e7` | 相同 | 通过 |
| 起始checkpoint SHA-256 | `02553b1ed7b8bbb5a1b1511134dbe06a60d7362923445908c5d6daca5c7ecabd` | 相同 | 通过 |
| SID registry SHA-256 | `5f12417ff5e60818753be8ec2d8c5dece350068a30a36d25f17bffab8dead29a` | 相同 | 通过 |
| 训练配置 | seed=17，steps=20，batch=64，样本=5000，lr=0.0003，AdamW | 相同 | 通过 |
| autocast enabled | true | true | 通过 |
| autocast dtype | bfloat16 | bfloat16 | 通过 |

## 3. 训练与保存恢复

| 指标 | GPU | NPU | 差异或判断 |
|---|---:|---:|---:|
| first loss | 5.594177 | 5.605469 | 绝对差0.011292 |
| last loss | 5.774569 | 5.787365 | 绝对差0.012797 |
| 20步平均loss | 5.707227 | 5.710366 | NPU-GPU=+0.003138 |
| 逐步loss平均绝对差 | — | — | 0.007879，即GPU平均loss的约0.1381% |
| 逐步loss最大绝对差 | — | — | 0.013265，发生在第15步 |
| loss差RMSE | — | — | 0.008705 |
| gradient L2 first | 0.809206 | 0.808553 | NPU低约0.08% |
| gradient L2 last | 0.929443 | 0.930359 | NPU高约0.10% |
| gradient max abs | 0.040527 | 0.042480 | NPU高约4.82% |
| gradient tensors | 80 | 80 | 一致 |
| all losses finite | true | true | 通过 |
| reload state match | true | true | 通过 |
| reload forward finite | true | true | 通过 |
| reload forward loss | 5.506104 | 5.497839 | 绝对差0.008265 |
| 最终状态 | passed | passed | 通过 |

## 4. 与各自FP32短训练的关系

| 平台 | FP32平均loss | BF16平均loss | BF16-FP32 |
|---|---:|---:|---:|
| GPU | 5.707635 | 5.707227 | -0.000408 |
| NPU | 5.707190 | 5.710366 | +0.003176 |

训练包含dropout且不同设备使用各自随机数实现，因此逐步loss和独立训练输出checkpoint不要求
完全相同。当前差异没有显示BF16造成非有限值、梯度异常或跨设备偏差明显放大。

## 5. 非正式运行观测

| 项目 | GPU | NPU |
|---|---:|---:|
| elapsed seconds | 1.0468 | 1.7937 |
| steps per second | 19.1067 | 11.1503 |
| peak allocated memory | 63.01 MiB | 86.00 MiB |

这些数值来自20步短训练，未隔离编译、预热和同步开销，也不保证不同后端显存统计语义完全
一致，因此只归档运行观测，不形成性能优劣结论。

## 6. 下一步

加载同一冻结Full checkpoint，在两台服务器使用固定validation cohort分别执行FP32与BF16，
比较v2列表指标和target/greedy/beam指纹。只有指标或离散输出差异异常时，才进入逐案例、
逐Module或逐API精度下钻。
