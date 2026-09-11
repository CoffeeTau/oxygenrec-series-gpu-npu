# NPU Stage-0服务器输出（截图转录版）

> 主日志：[`复现实验日志.md`](../../复现实验日志.md)  
> 环境快照：[`npu_server_environment_snapshot_2026-09-11.md`](../../../docs/npu_server_environment_snapshot_2026-09-11.md)  
> 来源：用户于2026-09-11回传的`environment.json`和`single_card.json`截图。  
> 边界：未收到服务器原始JSON文件；以下只转录截图可可靠辨认且影响验收判断的字段。

## environment关键字段

```text
collected_at_utc=2026-09-11T12:26:32.957516+00:00
platform=Linux-6.6.0-159.4.10.164.oe2403sp4.aarch64-aarch64-with-glibc2.36
machine=aarch64
cpu_count=192
python=3.11.0
torch=2.10.0
torch_runtime=2.10.0+cpu
torch_npu=2.10.0.post4.dev20260715
torch_npu_importable=True
npu_available=True
npu_device_count=8
distributed_hccl_available=True
device_name=Ascend950DT_9581
device_memory=93952MB
CANN_path=/usr/local/Ascend/cann-9.0.T550
npu-smi_returncode=0
atc_--version_returncode=255
atc_error=unknown command line flag 'version'
gcc=11.2.0
```

## single-card关键字段

```text
stage=npu_single_card
device=npu:0
device_name=Ascend950DT_9581
torch_version=2.10.0+cpu
torch_npu_version=2.10.0.post4.dev20260715
loss=93.5
gradient_l1=738.0
parameter_max_abs_delta=0.010100007057189941
checkpoint_match=True
checkpoint=checkpoints/npu_stage0/single_card.pt
```

## 原始判断

- Stage-0A设备与TorchNPU可见性通过；ATC精确版本未确认。
- Stage-0B单卡计算、反向、AdamW更新及checkpoint恢复通过。
- 本记录不包含OxygenREC模型forward、训练或GPU/NPU精度对齐证据。
