import hashlib
import json
import unittest

from public_smoke import encode, fixtures, png_rgb


class PublicFixtureTest(unittest.TestCase):
    def test_deterministic_and_append_only(self):
        cases, image = fixtures('fixture-model', 32)
        again, image_again = fixtures('fixture-model', 32)
        self.assertEqual(image, image_again)
        self.assertEqual(encode(cases), encode(again))
        self.assertEqual(cases[1][1], cases[2][1])
        self.assertTrue(image.startswith(b'\x89PNG\r\n\x1a\n'))
        self.assertNotEqual(hashlib.sha256(image).digest(), hashlib.sha256(png_rgb((255, 0, 0))).digest())
        messages = cases[1][1]['messages']
        self.assertEqual([m['role'] for m in messages], ['user', 'assistant', 'user', 'assistant', 'tool', 'user'])
        self.assertEqual(messages[3]['tool_calls'][0]['id'], messages[4]['tool_call_id'])
        self.assertEqual(messages[:1], cases[0][1]['messages'])
        self.assertEqual(cases[0][1]['tools'], cases[1][1]['tools'])
        self.assertFalse(cases[0][1]['chat_template_kwargs']['thinking'])
        self.assertEqual(json.loads(encode(cases)), json.loads(encode(again)))


if __name__ == '__main__':
    unittest.main()
