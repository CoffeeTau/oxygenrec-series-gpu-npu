# L20 server environment collection

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

