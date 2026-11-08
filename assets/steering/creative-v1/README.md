# Creative v1 — optional experimental activation directions

**Not enabled by default.** The normal startup serves the original official model. Loading this asset requires explicit flags on all four ranks and the paired frontend. It is a small behavior-steering asset, not another model checkpoint or a temperature preset.

## Contents and verification

- `directions.bin`: 819,216 bytes, `DSCR0001`, 40 × 5,120 little-endian float32 components.
- `directions.json`: readable values extracted losslessly from that binary, without private calibration labels or capture metadata.
- `manifest.json`: hashes, checkpoint compatibility, cache scope and default-off policy.
- `LICENSE`: MIT, retaining DeepSeek attribution.

Binary SHA256: `4ac401fd47fa275a9218bcfbc3d2521122221850fd7f6cf1437a90a518ed8a59`.

Recreate the exact binary in a **new output path** from the repository root:

```sh
python3 - <<'PY'
import json, subprocess
from pathlib import Path
p = Path('assets/steering/creative-v1')
m = json.loads((p / 'manifest.json').read_text())
subprocess.run(['python3', 'tools/pack_creative.py', str(p / 'directions.json'),
                '/tmp/creative-v1-repacked.bin', '--source-sha256', m['source_json_sha256'],
                '--checkpoint-revision', m['compatible_checkpoint_revision']], check=True)
PY
sha256sum /tmp/creative-v1-repacked.bin assets/steering/creative-v1/directions.bin
```

## Provenance and evidence limits

This is the project's existing legacy DeepSeek-V4.1 calibration asset. The available record describes 26 independent prompts, normalized pair-layer differences over layers 10–29, and an uncentered shared principal direction copied to 40 layers. It is not 40 independent behavioral directions. The exact activation-extraction checkpoint revision was not preserved in the available record. Do not describe it as newly fitted on the 1017 checkpoint. The original prompt/capture data is not distributed, so only **JSON-to-binary reproduction**, not complete refitting, is reproducible from this package.

It has been used with the official 1017 checkpoint and this runtime's optional mode path. Existing image/tool/cache/mode-switch checks qualify the integration, not a comprehensive behavioral score. No benchmark here establishes universal refusal reduction, better creativity or unchanged quality on every domain. Evaluate relevant tasks and retain regressions before adopting it.

## Enable only if you want it

Follow [the explicit steering setup](../../../docs/STEERING.md#explicitly-enable-the-bundled-asset-or-a-local-asset). Copy the binary to every node, use `tools/run.py` with `--enable-steering --creative-file /abs/directions.bin --creative-scope <manifest-suggested_scope>` on each rank, and `--enable-steering --creative-scope <same-scope>` on the frontend, alongside the normal startup arguments. The runtime then exposes `-normal` (strength 0) and `-creative` (strength 1.5) names. There is no per-request arbitrary-strength slider or live asset replacement. Directions and mode histories have separate cache identities.

To return to the default setup, drain the paired instance and start it with `tools/serve_original.py`; do not carry a live KV cache across that restart. Full model weights remain unchanged.

## 中文说明

这是**默认不加载**的实验性历史方向，二进制约0.82MB，不是完整模型权重。JSON公开全部数值，可逐字节复现二进制。私有提示标签和激活记录不公开；旧记录没有保留精确的采集模型修订，因此不能完整重做原拟合，也不能说是在1017上重新标定。

历史方法使用26个独立提示，汇总中层10～29的归一化配对差，提取未中心化共享主方向后复制到40层。已验证的加载、图片、工具、缓存及模式切换，不代表所有任务质量或拒答行为都得到改善。请先按自己的任务评估。

用户主动启用后可以按请求选Normal（强度0，原始路径）或Creative（强度1.5）。更换向量需要新的资产身份和配对实例重启；默认原始模型入口不会自动启用随附方向。详细方法和启动参数见[中文方案](../../../docs/STEERING.zh-CN.md)。
