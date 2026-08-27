# Qwen3-4B服务器下载与OxygenREC接入操作指南

> 适用环境：项目采集到的8×NVIDIA L20服务器。
>
> 当前目标：先验证冻结Qwen hidden state能够进入OxygenREC的Q2I/IGR接口。本轮不做SFT、RL、量化或多卡训练。
>
> 模型：`Qwen/Qwen3-4B-Instruct-2507`。

## 1. 进入项目和确认分支

```bash
cd /home/h50061831/oxygenrec-series-gpu-npu
pwd
git status --short
```

如果服务器上的实际项目目录不同，只执行第一条时替换成实际路径。不要在其他目录下载模型，否则默认测试脚本找不到权重。

## 2. 检查磁盘、GPU和Python环境

```bash
df -h .
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
python --version
python -c "import torch, transformers; print('torch', torch.__version__); print('transformers', transformers.__version__); print('cuda', torch.cuda.is_available(), torch.cuda.device_count())"
```

验收：

- 项目所在磁盘建议至少保留20 GiB空闲空间；官方BF16模型文件约8.06 GB，仍需给缓存、checkpoint和临时文件留余量。
- 至少一张L20有足够空闲显存；不要在已有训练任务占用的GPU上启动。
- `torch.cuda.is_available()`应为`True`。
- 当前服务器已采集到Transformers 4.57.6，满足Qwen3所需的4.51以上版本；不要为了本次测试随意升级PyTorch或CUDA。

## 3. 准备ModelScope下载工具

先检查：

```bash
python -c "import modelscope; print(modelscope.__version__)"
```

如果显示`ModuleNotFoundError: No module named 'modelscope'`，使用公司可访问/批准的PyPI源安装：

```bash
python -m pip install modelscope
```

安装后重新执行版本检查。不要升级现有PyTorch、CUDA或Transformers；本次只需要ModelScope的下载SDK。

## 4. 通过ModelScope直接下载到服务器项目目录

项目已提供下载脚本，它会：

1. 使用ModelScope上的Qwen官方仓库；
2. 下载指定的`master` revision；
3. 下载到项目的`models/`目录；
4. 检查配置、tokenizer、index和safetensors分片；
5. 写入`MODEL_SOURCE.json`保存provider、repo、revision和各权重分片大小；Hugging Face备用后端还会保存解析后的commit。

执行：

```bash
python scripts/download_qwen_model.py --provider modelscope
```

默认目标目录：

```text
models/Qwen3-4B-Instruct-2507/
```

下载中断后可以再次执行同一条命令。ModelScope会复用已经完成的文件并继续下载，不要手动删除半成品目录。

成功时最后一行类似：

```text
OK provider=modelscope model_dir=models/Qwen3-4B-Instruct-2507 revision=master shards=3 size_gib=...
```

ModelScope官方模型页：<https://modelscope.cn/qwen/Qwen3-4B-Instruct-2507>。

如果以后Hugging Face网络恢复，脚本仍保留备用后端：

```bash
python scripts/download_qwen_model.py --provider huggingface
```

## 5. 下载后进行只读完整性检查

```bash
ls -lh models/Qwen3-4B-Instruct-2507
du -sh models/Qwen3-4B-Instruct-2507
python -c "import json; print(json.load(open('models/Qwen3-4B-Instruct-2507/MODEL_SOURCE.json')))"
```

至少应看到：

```text
config.json
tokenizer.json
tokenizer_config.json
model.safetensors.index.json
model-00001-of-00003.safetensors
model-00002-of-00003.safetensors
model-00003-of-00003.safetensors
MODEL_SOURCE.json
```

不要修改模型目录中的`config.json`或tokenizer文件。

## 6. 执行单卡CUDA接入测试

先选择一张空闲GPU；下面以GPU 0为例：

```bash
CUDA_VISIBLE_DEVICES=0 bash run_server_test.sh
```

如果模型不在默认目录：

```bash
CUDA_VISIBLE_DEVICES=0 QWEN_MODEL_PATH=/实际模型目录 bash run_server_test.sh
```

脚本会依次完成：

1. 从本地目录加载Qwen3-4B backbone，不联网、不执行remote code；
2. 用官方chat template编码两条不同的行为证据prompt；
3. 冻结Qwen并提取最后层hidden state；
4. masked mean-pool并L2归一化为2560维特征；
5. 重复编码检查确定性；
6. 将特征输入OxygenREC instruction adapter；
7. 验证不同特征造成不同logits且adapter收到梯度；
8. 输出峰值已分配GPU显存。

成功输出应包含：

```text
OK device=cuda hidden_size=2560 tokens=(...) feature_delta=... \
feature_cosine=... determinism_error=... logit_delta=... \
adapter_grad=... peak_allocated_gib=...
```

验收标准：

- `hidden_size=2560`；
- `feature_delta>0`；
- `determinism_error<=1e-6`；
- `logit_delta>0`；
- `adapter_grad>0`；
- 没有CUDA OOM；
- `peak_allocated_gib`明显低于单卡约44.40 GiB。

`feature_cosine`用于观察两类行为prompt的区分程度，不设置人为通过阈值。

## 7. 常见问题

### 7.1 ModelScope连接超时

直接重新执行：

```bash
python scripts/download_qwen_model.py --provider modelscope
```

不要先删除模型目录。重复执行可复用已下载文件。

### 7.2 SSL证书验证失败

不要关闭模型权重下载的TLS验证。权重约8 GB，关闭验证会失去来源真实性和传输完整性保障。

优先处理顺序：

1. 确认服务器时间正确；
2. 使用公司批准的CA证书和代理配置；
3. 让运维修复容器/系统CA证书；
4. 在可验证的机器下载官方snapshot，再连同`MODEL_SOURCE.json`传到服务器。

如果因当前实验必须先下载，项目提供显式的临时关闭选项：

```bash
python scripts/download_qwen_model.py \
    --provider modelscope \
    --insecure-skip-tls-verify
```

该选项只在当前Python下载进程中令Requests使用`verify=False`，不会修改系统CA或其他shell命令。脚本会打印醒目warning，并在`MODEL_SOURCE.json`记录：

```json
"tls_verification": false
```

风险边界：代理可以查看或替换传输内容，下载来源真实性不能由TLS保证。只将它用于当前受控网络中的临时实验；取得公司CA后应恢复验证。

### 7.3 找不到模型目录

错误通常类似：

```text
local model directory not found
```

检查：

```bash
pwd
ls -ld models/Qwen3-4B-Instruct-2507
```

或者显式传入：

```bash
QWEN_MODEL_PATH=/绝对路径 bash run_server_test.sh
```

### 7.4 CUDA OOM

先确认是否有其他进程占用GPU：

```bash
nvidia-smi
```

本轮batch只有2、最大长度256，正常空闲L20不应OOM。若仍OOM，不要立即改成4-bit；先提供完整错误和`nvidia-smi`中允许公开的显存占用摘要。

### 7.5 FlashAttention或第三方扩展报错

本轮不强制使用FlashAttention 2，也不依赖bitsandbytes、DeepSpeed或FAISS。不要通过升级整个CUDA/PyTorch环境解决单个扩展错误；先保留完整错误信息，再决定是否显式切回PyTorch SDPA/eager attention。

## 8. 需要返回的最小信息

因公司信息安全限制，只需要提供最终`OK`行，或失败时提供：

- 报错类型和最后20行非敏感堆栈；
- 是否在模型加载、Qwen forward还是OxygenREC backward阶段失败；
- GPU型号和允许公开的空闲/峰值显存；
- 不要提供token、代理密码、内部地址或用户目录中的其他文件。

## 9. 本轮之后的工作

本轮通过后才进入：

1. 批量缓存RetailRocket样本的Qwen特征；
2. Qwen vs Generic vs Hash的matched Q2I/IGR消融；
3. 真实Qwen特征驱动的样例review；
4. 少量人工审核的结构化SFT数据；
5. 单卡BF16 LoRA SFT；
6. SFT收益成立后再进入偏好优化或SA-GCPO/RL。

不要跳过冻结特征验证直接开始RL。
