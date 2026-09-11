# OxygenREC服务器环境快照索引

已整理并长期保留两侧服务器快照：

- GPU：[`gpu_server_environment_snapshot_2026-08-21.md`](gpu_server_environment_snapshot_2026-08-21.md)，8×NVIDIA L20；
- NPU：[`npu_server_environment_snapshot_2026-09-11.md`](npu_server_environment_snapshot_2026-09-11.md)，8×Ascend 950DT。

以下命令用于重新采集GPU环境。NPU环境与单卡验证统一使用项目根目录的
`bash run_npu_stage0.sh`。

Copy `scripts/collect_server_env.py` to the server and run it inside the exact
Python environment that will train OxygenREC:

```bash
python collect_server_env.py --output oxygenrec_server_env.json
```

Return `oxygenrec_server_env.json`. The report contains:

- operating system, CPU count, and host-memory summary;
- Python executable and version;
- PyTorch build, CUDA runtime, cuDNN, NCCL, distributed backends;
- GPU model, compute capability, memory, and device count;
- `nvidia-smi` topology/NVLink output and `nvcc`/GCC versions;
- versions of Transformers, Accelerate, DeepSpeed, FlashAttention, Triton,
  FAISS, NumPy, and related packages;
- a strict allowlist of CUDA/NCCL/threading environment variables.

The script does not collect arbitrary environment variables, API keys, SSH
configuration, usernames, hostnames, GPU UUIDs, source code, or training data.
Review the JSON before sending it if the server has additional confidentiality
requirements.
