"""Exercise the actual launcher under a contaminated parent environment."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("runtime_launcher", ROOT / "tools/run.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class LaunchPolicy(unittest.TestCase):
    def test_original_cannot_inherit_steering(self):
        with tempfile.TemporaryDirectory() as d:
            args = ["run.py", "rank", "--rank", "1", "--fleet", d + "/fleet.conf", "--runtime-dir", d]
            parent = {"DSGB10_CREATIVE_FILE": "/untrusted/old.bin", "DSGB10_CREATIVE_SCOPE": "old",
                      "DSGB10_VISION_REVISION": "keep-vision", "PATH": os.environ.get("PATH", "")}
            with patch.object(sys, "argv", args), patch.dict(os.environ, parent, clear=True), \
                 patch.object(launcher.os, "execvpe", side_effect=SystemExit(0)) as child:
                with self.assertRaises(SystemExit) as done:
                    launcher.main()
                self.assertEqual(done.exception.code, 0)
                env = child.call_args.args[2]
                self.assertNotIn("DSGB10_CREATIVE_FILE", env)
                self.assertNotIn("DSGB10_CREATIVE_SCOPE", env)
                self.assertEqual(env["DSGB10_VISION_REVISION"], "keep-vision")

    def test_explicit_steering_uses_explicit_asset(self):
        with tempfile.TemporaryDirectory() as d:
            args = ["run.py", "rank", "--rank", "1", "--fleet", d + "/fleet.conf", "--runtime-dir", d,
                    "--enable-steering", "--creative-file", d + "/mine.bin", "--creative-scope", "my-revision:sha"]
            with patch.object(sys, "argv", args), patch.object(launcher.os, "execvpe", side_effect=SystemExit(0)) as child:
                with self.assertRaises(SystemExit):
                    launcher.main()
                self.assertEqual(child.call_args.args[2]["DSGB10_CREATIVE_SCOPE"], "my-revision:sha")
                self.assertEqual(Path(child.call_args.args[2]["DSGB10_CREATIVE_FILE"]), Path(d) / "mine.bin")

    def test_original_entry_rejects_steering(self):
        for flag in ("--enable-steering", "--creative-file=old.bin", "--creative-scope=old"):
            result = subprocess.run([sys.executable, str(ROOT / "tools/serve_original.py"), flag], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"does not accept steering", result.stderr)

    def test_asset_requires_opt_in(self):
        with tempfile.TemporaryDirectory() as d:
            result = subprocess.run([sys.executable,str(ROOT/'tools/run.py'),'rank',
                      '--rank','0','--fleet',d+'/fleet','--runtime-dir',d,
                      '--creative-file',d+'/vector.bin','--creative-scope','asset'],capture_output=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn(b'require --enable-steering',result.stderr)

    def test_steered_frontend_requires_scope(self):
        with tempfile.TemporaryDirectory() as d:
            result = subprocess.run([sys.executable,str(ROOT/'tools/run.py'),'frontend',
                      '--runtime-dir',d,'--enable-steering'],capture_output=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn(b'requires --creative-scope',result.stderr)


if __name__ == "__main__":
    unittest.main()
