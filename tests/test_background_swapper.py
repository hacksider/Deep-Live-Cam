import unittest
from unittest.mock import patch
import numpy as np

import modules.globals
from modules.processors.frame import background_swapper


class DummyFace:
    def __init__(self, bbox=None, kps=None, pose=None):
        self.bbox = np.array(bbox, dtype=np.float32) if bbox is not None else None
        self.kps = np.array(kps, dtype=np.float32) if kps is not None else None
        self.pose = np.array(pose, dtype=np.float32) if pose is not None else None


class TestBackgroundSwapper(unittest.TestCase):
    def setUp(self):
        modules.globals.background_swap = True
        modules.globals.background_path = None
        background_swapper._CACHED_BG_PATH = None
        background_swapper._CACHED_BG_IMAGE = None
        background_swapper._SMOOTHED_POV = None

    def test_compute_viewer_pov_none_face(self):
        pov = background_swapper.compute_viewer_pov(None, (480, 640))
        self.assertEqual(len(pov), 5)
        self.assertEqual(pov, (0.0, 0.0, 0.0, 0.0, 0.0))

    def test_compute_viewer_pov_with_face_bbox_and_pose(self):
        # Face offset to the right and tilted
        face = DummyFace(bbox=[400, 200, 600, 400], pose=[10.0, -15.0, 0.0])
        dx, dy, yaw, pitch, roll = background_swapper.compute_viewer_pov(face, (480, 640))
        # Center x is 320, face center x is 500, so dx > 0
        self.assertGreater(dx, 0.0)
        self.assertEqual(yaw, -15.0)
        self.assertEqual(pitch, 10.0)

    def test_render_3d_perspective_background(self):
        bg = np.ones((480, 640, 3), dtype=np.uint8) * 120
        warped = background_swapper.render_3d_perspective_background(
            bg, (480, 640), dx=0.5, dy=-0.2, yaw=15.0, pitch=-10.0
        )
        self.assertEqual(warped.shape, (480, 640, 3))
        self.assertEqual(warped.dtype, np.uint8)
        self.assertGreater(warped.max(), 0)

    @patch("modules.processors.frame.background_swapper.get_one_face")
    @patch("modules.processors.frame.background_swapper.predict_matte")
    def test_swap_background_blending(self, mock_predict_matte, mock_get_one_face):
        mock_get_one_face.return_value = None
        # Matte with 1.0 (foreground) in center and 0.0 (background) elsewhere
        matte = np.zeros((200, 200), dtype=np.float32)
        matte[50:150, 50:150] = 1.0
        mock_predict_matte.return_value = matte

        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        frame[50:150, 50:150] = 255

        bg = np.full((200, 200, 3), 100, dtype=np.uint8)
        background_swapper._CACHED_BG_PATH = "dummy.png"
        background_swapper._CACHED_BG_IMAGE = bg
        modules.globals.background_path = "dummy.png"

        result = background_swapper.swap_background(frame)
        self.assertEqual(result.shape, (200, 200, 3))
        self.assertEqual(result.dtype, np.uint8)
        # Foreground person area should retain original intensity
        self.assertEqual(result[100, 100, 0], 255)
        # Background area should match background
        self.assertEqual(result[10, 10, 0], 100)

    def test_process_frame_toggle_disabled(self):
        modules.globals.background_swap = False
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 42
        result = background_swapper.process_frame(None, frame)
        np.testing.assert_array_equal(result, frame)


if __name__ == "__main__":
    unittest.main()
