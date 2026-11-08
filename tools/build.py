#!/usr/bin/env python3
"""Build the complete paired C4 source in new directories, without using a GPU.

The global CUDA/Rust toolchain and downloaded source dependency caches may be
reused. No previous engine archive, object or frontend executable is an input.
This script does not load weights, contact a serving endpoint or start ranks.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "ds41f-4xgx10-switchless"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build-dir", type=Path, default=ROOT / "build")
    p.add_argument("--jobs", type=int, default=2)
    p.add_argument("--cargo", default=shutil.which("cargo") or "cargo")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--host-only", action="store_true")
    p.add_argument("--full-frontend-tests", action="store_true",
                   help="run ignored transport/tokenizer tests too; requires DSGB10_TEST_TOKENIZER")
    a = p.parse_args()
    if not 1 <= a.jobs <= 8:
        p.error("--jobs must be between 1 and 8")
    if a.full_frontend_tests and (a.host_only or not Path(os.environ.get("DSGB10_TEST_TOKENIZER", "")).is_file()):
        p.error("--full-frontend-tests needs a real DSGB10_TEST_TOKENIZER file and a full build")
    out = a.build_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="", CARGO_TARGET_DIR=str(out / "rust"))
    result = {"schema": 1, "started": time.time(), "passed": False,
              "gpu_executed": False, "clean_build": True, "commands": [],
              "source_files": {}, "artifacts": {},
              "entrypoint_sha256": digest(__file__),
              "profile_sha256": digest(ROOT / "profiles/c4.json"),
              "full_frontend_tests": a.full_frontend_tests}
    for folder in [CONFIG / "engine", CONFIG / "frontend", CONFIG / "vision"]:
        for f in sorted(folder.rglob("*")):
            if f.is_file() and not any(x in ("build", "target", "__pycache__") for x in f.relative_to(folder).parts):
                result["source_files"][f.relative_to(ROOT).as_posix()] = digest(f)

    def run(args, cwd=ROOT, timeout=1800):
        args = list(map(str, args))
        print("RUN " + json.dumps(args), flush=True)
        t = time.time()
        r = subprocess.run(args, cwd=cwd, env=env, timeout=timeout)
        result["commands"].append({"argv": args, "seconds": time.time() - t, "exit": r.returncode})
        if r.returncode:
            raise RuntimeError("command failed with exit " + str(r.returncode))

    try:
        run(["cmake", "-S", CONFIG / "engine/tests", "-B", out / "host", "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_TESTING=ON"])
        run(["cmake", "--build", out / "host", "--parallel", a.jobs])
        run(["ctest", "--test-dir", out / "host", "--output-on-failure"])
        if not a.host_only:
            run(["cmake", "-S", CONFIG / "engine", "-B", out / "engine", "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_TESTING=OFF"])
            run(["cmake", "--build", out / "engine", "--parallel", a.jobs,
                 "--target", "dsgb10_rank", "test_native24", "creative_test", "vision_test", "spec_batch_test"])
            opts = ["--locked"] + (["--offline"] if a.offline else [])
            test_args = ["--", "--include-ignored", "--test-threads=1"] if a.full_frontend_tests else []
            run([a.cargo, "test", *opts, "--jobs", a.jobs, "--lib", "--tests", *test_args], CONFIG / "frontend")
            run([a.cargo, "build", *opts, "--jobs", a.jobs, "--release", "--bins"], CONFIG / "frontend")
            for name, f in {
                "engine": out / "engine/server/dsgb10_rank",
                "frontend": out / "rust/release/dsfront",
                "native24_probe": out / "engine/kernels/test_native24",
                "creative_probe": out / "engine/model/creative_test",
                "vision_probe": out / "engine/model/vision_test",
            }.items():
                result["artifacts"][name] = {"path": str(f), "sha256": digest(f), "bytes": f.stat().st_size}
            run([out / "engine/model/spec_batch_test"])
        result["passed"] = True
    except BaseException as e:
        result["error"] = type(e).__name__ + ": " + str(e)
        raise
    finally:
        result["finished"] = time.time()
        (out / "BUILD_RESULT.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in result.items() if k not in ("source_files", "commands")}), flush=True)


if __name__ == "__main__":
    main()
