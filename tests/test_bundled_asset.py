"""Check the distributed asset's bytes, readable representation and opt-in policy."""
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "assets/steering/creative-v1"


class BundledAssetTest(unittest.TestCase):
    def test_manifest_and_readable_binary_reproduction(self):
        m = json.loads((ASSET / "manifest.json").read_text())
        raw = (ASSET / "directions.bin").read_bytes()
        source = (ASSET / "directions.json").read_bytes()
        self.assertFalse(m["default_enabled"])
        self.assertEqual(len(raw), m["bytes"])
        self.assertEqual(hashlib.sha256(raw).hexdigest(), m["binary_sha256"])
        self.assertEqual(hashlib.sha256(source).hexdigest(), m["source_json_sha256"])
        self.assertEqual(raw[:16], b"DSCR0001" + struct.pack("<II", 40, 5120))
        vectors = json.loads(source)["directions"]
        for i in range(40):
            r = vectors[str(i)]
            self.assertEqual(len(r), 5120)
            self.assertTrue(all(math.isfinite(x) for x in r))
            self.assertLess(abs(math.fsum(x*x for x in r)-1), 2e-4)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "directions.bin"
            subprocess.run([sys.executable, str(ROOT / "tools/pack_creative.py"),
                            str(ASSET / "directions.json"), str(output),
                            "--source-sha256", m["source_json_sha256"],
                            "--checkpoint-revision", m["compatible_checkpoint_revision"]],
                           check=True, capture_output=True)
            self.assertEqual(output.read_bytes(), raw)
            packed = json.loads(output.with_suffix(".bin.json").read_text())
            self.assertEqual(packed["suggested_scope"], m["suggested_scope"])
        self.assertIn("Permission is hereby granted", (ASSET / "LICENSE").read_text())
        self.assertIn("2023 DeepSeek", (ASSET / "LICENSE").read_text())


if __name__ == "__main__":
    unittest.main()
