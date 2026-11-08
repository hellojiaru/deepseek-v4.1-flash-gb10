# Prepare the official model

This engine serves `deepseek-ai/DeepSeek-V4.1-Flash-1017`, revision `1bae0e1088993b9b82de08f59de5be5a34e42170`. Download it using your normal Hugging Face workflow, subject to its license. The repository contains no model weights. An NVFP4 or EXL3 checkpoint is not a drop-in replacement.

The converter preserves the checkpoint's native mixed precision and creates the engine's tensor layout; it does not apply another quantization. Each node needs its own rank's DSW file, Engram shards and the shared hash map. The full checkpoint is large: plan storage for the download and converted files before starting.

After the [clean build](BUILD_AND_RUN.md), build the preparation tools:

```sh
cmake --build /opt/deepseek-build/engine --parallel 2 \
  --target dsw_convert dsw_verify dsw_load engram_pack
```

For each rank, use a new output directory. Example for rank 1:

```sh
mkdir -p /models/deepseek-rank1
/opt/deepseek-build/engine/weights/dsw_convert \
  --ckpt /models/DeepSeek-V4.1-Flash-1017 --rank 1 \
  --out /models/deepseek-rank1/rank1.dsw
/opt/deepseek-build/engine/weights/dsw_verify \
  --ckpt /models/DeepSeek-V4.1-Flash-1017 \
  --file /models/deepseek-rank1/rank1.dsw --threads 4
/opt/deepseek-build/engine/engram/engram_pack \
  /models/DeepSeek-V4.1-Flash-1017 /models/deepseek-rank1 1
python3 ds41f-4xgx10-switchless/engine/engram/py/make_hashmap_stdlib.py \
  /models/DeepSeek-V4.1-Flash-1017 /models/deepseek-rank1/engram_hashmap.bin
```

Conversion uses a GPU; perform it while the target is reserved, without another full model resident. Repeat for ranks 0, 2 and 3, and copy each rank's files to its node. Each rank's DSW is approximately 72.4 GiB and the two local Engram shards approximately 47.3 GiB; conversion and revision can affect exact sizes. These are storage sizes, not additional KV windows. Keep the original checkpoint files needed by the vision worker and tokenizer on rank 0.

Copy [fleet/homelab.conf](../ds41f-4xgx10-switchless/fleet/homelab.conf) to your private configuration. Set `RANKn_WEIGHTS`, `RANKn_ENGRAM`, `RANKn_HASHMAP`, `RANKn_SPILL`, the actual RDMA devices/GID and the addresses of both ring neighbours. Do not reuse its example hostnames/IPs as your network configuration.

The [layout documentation](../ds41f-4xgx10-switchless/docs/layout.md) explains the byte checks and rank identity. The source tree retains historical setup scripts for provenance; use the root launchers and profiles for this project's current defaults.
