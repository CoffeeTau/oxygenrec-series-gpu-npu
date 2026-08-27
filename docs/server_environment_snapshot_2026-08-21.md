# OxygenREC GPU 服务器环境快照

> 信息来源：项目根目录 `GPU信息.pdf` 中的6页截图。
>
> 原始环境采集时间：`2026-08-21T16:51:40.986654+00:00`。
>
> 整理时间：`2026-08-27 10:55:45 CST`。
>
> 用途：在原PDF删除后保留模型选择、训练配置和环境兼容性判断所需的信息。截图未完整展示的内容明确标为“未确认”，不作推测。

## 1. 系统与主机资源

| 项目 | 采集值 |
|---|---|
| 操作系统 | Linux |
| Platform | `Linux-5.15.0-25-generic-x86_64-with-glibc2.39` |
| Kernel release | `5.15.0-25-generic` |
| 架构 | `x86_64` |
| CPU逻辑核数 | 176 |
| 主机内存总量 | `924332060 kB`，约881.51 GiB |
| 采集时可用内存 | `823530604 kB`，约785.38 GiB |
| Swap总量 | `12582904 kB`，约12.00 GiB |
| 采集时空闲Swap | `2977456 kB`，约2.84 GiB |

## 2. Python环境

| 项目 | 采集值 |
|---|---|
| Python | `3.12.3` |
| 实现 | CPython |
| 编译器标记 | GCC 13.3.0 |
| Python可执行文件 | `/usr/bin/python` |
| Prefix | `/usr` |

## 3. PyTorch与CUDA运行时

| 项目 | 采集值 |
|---|---|
| PyTorch | `2.11.0a0+a6c236b9fd.nv26.3.46836102` |
| torchvision | `0.25.0a0+b7d91027.nv26.3.46836102` |
| torchaudio | 未安装或采集结果为`null` |
| PyTorch CUDA runtime | `13.2` |
| CUDA available | `True` |
| CUDA device count | 8 |
| PyTorch distributed | available |
| NCCL backend | available |
| Gloo backend | available |
| cuDNN version | `92000` |
| cuDNN available | `True` |
| NCCL version | `2.29.7` |
| CUDA_HOME | `/usr/local/cuda` |
| NVIDIA_VISIBLE_DEVICES | `all` |
| PyTorch编译架构 | `sm_75, sm_80, sm_86, sm_90, sm_100, sm_120, compute_120` |

说明：截图显示PyTorch版本为NVIDIA容器风格的开发版/预发布构建，不应按普通PyPI稳定版假设兼容性。安装额外CUDA扩展前必须在该环境单独编译或验证。

## 4. GPU

共采集到8张配置一致的GPU（index 0-7）：

| 项目 | 每张GPU的采集值 |
|---|---|
| 型号 | NVIDIA L20 |
| Compute capability | 8.9 |
| 显存字节数 | `47677177856` bytes |
| 二进制容量 | 约44.40 GiB |
| `nvidia-smi`容量 | `46068 MiB` |
| Streaming multiprocessors | 92 |
| Driver version | `570.172.08` |

容量说明：设备通常按十进制标称为48 GB；PyTorch截图中的`total_memory_bytes`折算后约44.40 GiB。训练预算应使用实际可分配显存并预留CUDA context、通信和碎片空间，不能按48 GiB全部占满。

## 5. 机器学习软件包

| 包 | 版本/状态 |
|---|---|
| transformers | `4.57.6` |
| accelerate | `1.14.0` |
| DeepSpeed | 未安装 |
| flash-attn | `2.7.4.post1+git...`；截图中的完整git后缀未可靠辨认 |
| xformers | 未安装 |
| Triton | `3.6.0+git5d72932fc5.nv26.3` |
| bitsandbytes | 未安装 |
| Apex | `0.1` |
| faiss-cpu | 未安装 |
| faiss-gpu | 未安装 |
| NumPy | `1.26.4` |
| SciPy | `1.17.1` |
| scikit-learn | `1.5.1` |
| datasets | `3.6.0` |
| sentencepiece | `0.2.2` |
| safetensors | `0.7.0` |

## 6. 系统工具与截图边界

| 项目 | 状态 |
|---|---|
| `nvidia-smi` | 可用，return code 0 |
| GPU topology命令 | 可用，return code 0；截图只保留表头/部分转义内容，无法重建GPU间连接矩阵 |
| NVLink status命令 | return code 0，stdout为空；仅记录该现象，不据此断言硬件一定不存在NVLink |
| `nvcc --version` | 可用，return code 0；完整版本字符串在截图右侧被截断，未确认 |
| `gcc --version` | GCC 13.3.0（Ubuntu 13.3.0-6ubuntu2~24.04.1） |

## 7. 对后续Qwen/SFT/RL工作的直接约束

1. 单卡实际约44.40 GiB，可优先进行小中型Qwen的BF16推理、冻结特征提取或LoRA/QLoRA验证；具体规格仍需按序列长度、batch、optimizer和是否同时驻留reference/reward模型做显存实测。
2. 8卡均被PyTorch识别，NCCL backend可用，但现有证据只说明“接口可用”，尚未完成NCCL多卡通信和训练稳定性验证。
3. 当前已有Transformers、Accelerate、FlashAttention、SentencePiece和Safetensors，具备接入Hugging Face/Qwen权重的基础。
4. 当前没有DeepSpeed、bitsandbytes和FAISS。不能直接假设ZeRO、4-bit QLoRA或向量索引脚本可运行；需要时应先做独立安装与兼容性测试。
5. PyTorch/CUDA版本较新且属于NVIDIA定制构建，第三方二进制wheel可能不兼容。优先使用纯PyTorch/Transformers路径，再逐项引入量化和自定义CUDA扩展。
6. RL阶段必须分别核算actor、reference、reward/teacher、rollout KV cache、optimizer state和activation，不能用SFT单模型显存估算替代。

## 8. 删除原PDF后的信息边界

本文保留了截图中对模型选择和训练路线有用、且能够可靠辨认的字段。以下内容没有被完整保留：

- 8卡拓扑矩阵及GPU间PCIe/NVLink关系；
- NVCC完整版本字符串；
- FlashAttention完整git revision；
- 容器镜像名称、磁盘容量、网络配置和实际空闲GPU显存；
- NCCL多卡实测结果。

如果后续任务依赖上述字段，应重新运行项目中的`collect_server_env.py`或执行针对性检查，不能从本文推断。
