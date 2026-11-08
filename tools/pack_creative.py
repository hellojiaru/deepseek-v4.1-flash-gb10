#!/usr/bin/env python3
"""Pack an externally supplied, SHA-pinned steering JSON into DSCR0001.

This converts an existing asset; it does not train, derive or download directions.
The format matches the current C4 loader and the previous deployment converter.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--source-sha256", required=True)
    p.add_argument("--checkpoint-revision", required=True)
    a = p.parse_args()
    raw = a.source.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != a.source_sha256.lower():
        p.error("source SHA256 mismatch")
    asset = json.loads(raw)
    directions = asset["directions"]
    if asset["hidden_size"] != 5120 or set(directions) != {str(i) for i in range(40)}:
        p.error("expected 40 layers with hidden size 5120")
    for values in directions.values():
        if len(values) != 5120 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in values):
            p.error("invalid direction shape or nonfinite value")
        if abs(math.fsum(float(x) * float(x) for x in values) - 1.0) > 2e-4:
            p.error("each layer direction must have unit squared norm (within 2e-4), as required by the engine")
    data = b"DSCR0001" + struct.pack("<II", 40, 5120)
    data += b"".join(struct.pack("<5120f", *directions[str(i)]) for i in range(40))
    meta = a.output.with_suffix(a.output.suffix + ".json")
    if a.output.exists() or meta.exists():
        p.error("output or manifest already exists; choose a new owned path")
    manifest = dict(format="DSCR0001", layers=40, hidden_size=5120,
                    source_sha256=sha, binary_sha256=hashlib.sha256(data).hexdigest(),
                    bytes=len(data), checkpoint_revision=a.checkpoint_revision,
                    suggested_scope=a.checkpoint_revision + ":" + sha + ":lambda1.5:v1")
    with a.output.open("xb") as f:
        f.write(data)
    with meta.open("x", encoding="utf-8") as f:
        f.write(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
