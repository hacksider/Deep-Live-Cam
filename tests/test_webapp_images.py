import unittest

import numpy as np

from webapp.images import decode_image, encode_jpeg


class ImageTests(unittest.TestCase):
    def test_garbage_bytes_decode_to_none(self):
        self.assertIsNone(decode_image(b"not a jpeg"))

    def test_encode_then_decode_roundtrip(self):
        frame = np.full((16, 24, 3), 128, dtype=np.uint8)
        data = encode_jpeg(frame)
        self.assertIsInstance(data, bytes)
        decoded = decode_image(data)
        self.assertIsNotNone(decoded)
        self.assertEqual(decoded.shape, (16, 24, 3))


if __name__ == "__main__":
    unittest.main()
