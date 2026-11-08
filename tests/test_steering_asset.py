import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

TOOL = Path(__file__).resolve().parents[1] / 'tools/pack_creative.py'


class SteeringAsset(unittest.TestCase):
    def test_valid_asset_and_non_unit_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            asset = dict(hidden_size=5120, directions={str(i): [1.0]+[0.0]*5119 for i in range(40)})
            for valid in (True, False):
                if not valid:
                    asset['directions']['12'][0] = 2.0
                source = root / ('valid.json' if valid else 'bad.json')
                source.write_text(json.dumps(asset), encoding='utf-8')
                output = source.with_suffix('.bin')
                result = subprocess.run([sys.executable,str(TOOL),str(source),str(output),
                         '--source-sha256',hashlib.sha256(source.read_bytes()).hexdigest(),
                         '--checkpoint-revision','test-revision'],capture_output=True)
                if valid:
                    self.assertEqual(result.returncode,0,result.stderr)
                    self.assertEqual(output.read_bytes()[:8],b'DSCR0001')
                    self.assertEqual(output.stat().st_size,16+40*5120*4)
                else:
                    self.assertNotEqual(result.returncode,0)
                    self.assertIn(b'unit squared norm',result.stderr)
                    self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
