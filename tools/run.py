#!/usr/bin/env python3
"""Run one component from a clean C4 build. Fleet reservation is external.

Start ranks 1..3, rank0, then the frontend on the chosen hosts. A running full
model must be drained/stopped first: this launcher never stops other services.
Secrets are read by the frontend from --key-file, never printed or copied here.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("component", choices=["rank", "frontend", "vision"])
    p.add_argument("--profile", choices=["c4", "c8-reference"], default="c4")
    p.add_argument("--build-dir", type=Path, default=ROOT / "build")
    p.add_argument("--runtime-dir", type=Path, required=True)
    p.add_argument("--fleet", type=Path)
    p.add_argument("--rank", type=int, choices=range(4))
    p.add_argument("--spill-dir", type=Path)
    p.add_argument("--disk-tag", default="local-c4")
    p.add_argument("--listen", default="127.0.0.1:18893")
    p.add_argument("--key-file", type=Path)
    p.add_argument("--tokenizer", type=Path)
    p.add_argument("--model-name", default="deepseek-v4.1-flash-dsgb10-official-1017")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--revision", default="1bae0e1088993b9b82de08f59de5be5a34e42170")
    p.add_argument("--vision-socket", type=Path)
    p.add_argument("--vision-revision")
    p.add_argument("--creative-file", type=Path)
    p.add_argument("--creative-scope")
    p.add_argument("--enable-steering", action="store_true",
                   help="opt in to request-selectable activation steering; off by default")
    p.add_argument("--print-command", action="store_true")
    a = p.parse_args()
    profile = json.loads((ROOT / "profiles" / (a.profile + ".json")).read_text(encoding="utf-8"))
    if not a.enable_steering and (a.creative_file or a.creative_scope):
        p.error("steering assets require --enable-steering; the default serves the original model")
    if a.enable_steering:
        if a.component == "vision":
            p.error("steering belongs to rank/frontend, not the vision encoder")
        if not a.creative_scope or (a.component == "rank" and not a.creative_file):
            p.error("steering requires --creative-scope and, for rank, --creative-file")
    rt = a.runtime_dir.resolve()
    rt.mkdir(parents=True, exist_ok=True, mode=0o700)
    env = os.environ.copy()
    # A shell or service may retain a previous deployment's steering settings.
    # Only explicit command-line opt-in can enable them for this process.
    for key in ("DSGB10_CREATIVE_FILE", "DSGB10_CREATIVE_SCOPE"):
        env.pop(key, None)
    for key, value in [("DSGB10_CREATIVE_FILE", a.creative_file), ("DSGB10_CREATIVE_SCOPE", a.creative_scope),
                       ("DSGB10_VISION_SOCKET", a.vision_socket), ("DSGB10_VISION_REVISION", a.vision_revision)]:
        if value is not None:
            env[key] = str(value)
    build = a.build_dir.resolve()
    if a.component == "rank":
        if a.rank is None or not a.fleet:
            p.error("rank requires --rank and --fleet")
        if bool(a.creative_file) != bool(a.creative_scope):
            p.error("pass both --creative-file and --creative-scope")
        cmd = [str(build / "engine/server/dsgb10_rank"), "--fleet", str(a.fleet.resolve()), "--rank", str(a.rank),
               "--socket", str(rt / "engine.sock"), "--disk-tag", a.disk_tag, *profile["engine_args"]]
        if a.spill_dir:
            a.spill_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            cmd += ["--disk-dir", str(a.spill_dir.resolve())]
    elif a.component == "frontend":
        if not a.key_file or not a.tokenizer or not a.vision_socket or not a.vision_revision:
            p.error("frontend requires --key-file, --tokenizer, --vision-socket and --vision-revision")
        cmd = [str(build / "rust/release/dsfront"), "serve", "--listen", a.listen,
               "--engine-socket", str(rt / "engine.sock"), "--admin-socket", str(rt / "admin.sock"),
               "--api-key-file", str(a.key_file.resolve()), "--tokenizer", str(a.tokenizer.resolve()),
               "--model-name", a.model_name, *profile["frontend_args"]]
    else:
        if not a.checkpoint:
            p.error("vision requires --checkpoint")
        cmd = [sys.executable, "-u", str(ROOT / "ds41f-4xgx10-switchless/vision/encoder.py"),
               "--checkpoint", str(a.checkpoint.resolve()), "--revision", a.revision,
               "--socket", str(a.vision_socket or rt / "vision.sock")]
    if a.print_command:
        print(json.dumps({"argv": cmd, "environment_names": [k for k in env if k.startswith("DSGB10_")]}, indent=2))
        return
    os.execvpe(cmd[0], cmd, env)


if __name__ == "__main__":
    main()
