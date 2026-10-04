import unittest
from types import SimpleNamespace

from webapp.look import default_look, make_renderer, parse_look


def _globals():
    return SimpleNamespace(
        opacity=0.25,
        sharpness=0.0,
        mouth_mask=False,
        mouth_mask_size=0.0,
        show_mouth_mask_box=True,
        many_faces=False,
        poisson_blend=False,
        enable_interpolation=True,
        fp_ui={
            "face_enhancer": False,
            "face_enhancer_gpen256": False,
            "face_enhancer_gpen512": False,
        },
    )


def _look(**overrides):
    look = default_look()
    look.update(overrides)
    return look


class ParseLookTests(unittest.TestCase):
    def test_defaults_and_bounds(self):
        look, error = parse_look(_look(sharpness=5, mouth=100, opacity=0))
        self.assertIsNone(error)
        self.assertEqual(look["sharpness"], 5.0)
        self.assertEqual(look["mouth"], 100.0)
        self.assertEqual(look["opacity"], 0.0)

    def test_rejects_out_of_range_bools_and_unknown_enhancers(self):
        for payload in (
            _look(opacity=1.1),
            _look(sharpness=-0.1),
            _look(mouth=101),
            _look(opacity=True),
            _look(many_faces=1),
            _look(enhancer="GFPGAN"),
            {"opacity": 1},
        ):
            look, error = parse_look(payload)
            self.assertIsNone(look)
            self.assertEqual(error, "bad look")


class RendererTests(unittest.TestCase):
    def test_each_call_uses_its_look_and_restores_globals(self):
        globals_module = _globals()
        original_flags = globals_module.fp_ui
        seen = []

        def swap(source, face, frame):
            seen.append(
                (
                    globals_module.opacity,
                    globals_module.sharpness,
                    globals_module.mouth_mask,
                    globals_module.mouth_mask_size,
                    globals_module.many_faces,
                    globals_module.poisson_blend,
                    globals_module.enable_interpolation,
                    globals_module.fp_ui["face_enhancer"],
                    face.bbox,
                )
            )
            return frame + b"+"

        render = make_renderer(
            globals_module=globals_module,
            detect_one=lambda frame: SimpleNamespace(bbox=(1, 2, 3, 4)),
            detect_many=lambda frame: [
                SimpleNamespace(bbox=(1, 1, 2, 2)),
                SimpleNamespace(bbox=(3, 3, 4, 4)),
            ],
            swap=swap,
            post=lambda frame, boxes: (frame, boxes),
            landmarks=lambda frame, faces: seen.append("landmarks"),
            enhance=lambda name, frame, faces: seen.append(name) or frame,
        )
        result = render(b"src", b"frame", _look(opacity=0.8, sharpness=2, mouth=40, enhancer="gfpgan"))
        self.assertEqual(seen[0], "landmarks")
        self.assertEqual(seen[1], "gfpgan")
        self.assertEqual(seen[2][0], 0.8)
        self.assertEqual(seen[2][1], 2)
        self.assertTrue(seen[2][2])
        self.assertEqual(seen[2][3], 40)
        self.assertFalse(seen[2][6])
        self.assertTrue(seen[2][7])
        self.assertEqual(result[0], b"frame+")
        many = render(b"src", b"frame", _look(many_faces=True, poisson=True))
        self.assertEqual(len(many[1]), 2)
        self.assertEqual(globals_module.opacity, 0.25)
        self.assertFalse(globals_module.mouth_mask)
        self.assertTrue(globals_module.show_mouth_mask_box)
        self.assertTrue(globals_module.enable_interpolation)
        self.assertFalse(globals_module.fp_ui["face_enhancer"])
        self.assertFalse(original_flags["face_enhancer"])

    def test_no_face_skips_swap_and_still_restores(self):
        globals_module = _globals()
        called = []
        render = make_renderer(
            globals_module=globals_module,
            detect_one=lambda frame: None,
            detect_many=lambda frame: None,
            swap=lambda source, face, frame: called.append("swap") or frame,
            post=lambda frame, boxes: frame,
            landmarks=lambda frame, faces: called.append("landmarks"),
            enhance=lambda name, frame, faces: frame,
        )
        self.assertIsNone(render(b"src", b"frame", _look()))
        self.assertEqual(called, [])
        self.assertEqual(globals_module.opacity, 0.25)

    def test_enhancer_failure_still_swaps_and_restores(self):
        globals_module = _globals()

        def enhance(name, frame, faces):
            raise RuntimeError("no model")

        render = make_renderer(
            globals_module=globals_module,
            detect_one=lambda frame: SimpleNamespace(bbox=(0, 0, 1, 1)),
            detect_many=lambda frame: None,
            swap=lambda source, face, frame: b"swapped",
            post=lambda frame, boxes: frame,
            landmarks=lambda frame, faces: None,
            enhance=enhance,
        )
        with self.assertLogs("webapp.look", level="ERROR"):
            result = render(b"src", b"frame", _look(enhancer="gpen256", opacity=0.2))
        self.assertEqual(result, b"swapped")
        self.assertEqual(globals_module.opacity, 0.25)
        self.assertFalse(globals_module.fp_ui["face_enhancer_gpen256"])

    def test_swap_failure_restores_globals(self):
        globals_module = _globals()

        def explode(source, face, frame):
            raise RuntimeError("swap failed")

        render = make_renderer(
            globals_module=globals_module,
            detect_one=lambda frame: SimpleNamespace(bbox=(0, 0, 1, 1)),
            detect_many=lambda frame: None,
            swap=explode,
            post=lambda frame, boxes: frame,
            landmarks=lambda frame, faces: None,
            enhance=lambda name, frame, faces: frame,
        )
        with self.assertRaises(RuntimeError):
            render(b"src", b"frame", _look(opacity=0.2))
        self.assertEqual(globals_module.opacity, 0.25)
        self.assertTrue(globals_module.enable_interpolation)


if __name__ == "__main__":
    unittest.main()
