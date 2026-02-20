"""Tests for modules/face_analyser.py — unit-testable behaviour."""
import pytest
from unittest.mock import patch


def test_default_target_face_empty_map():
    """default_target_face must not crash when source_target_map is empty."""
    import modules.globals
    with patch.object(modules.globals, 'source_target_map', []):
        from modules import face_analyser
        face_analyser.default_target_face()


def test_default_target_face_no_faces_in_frame():
    """default_target_face must not crash when no frame has faces (best_face stays None)."""
    import modules.globals
    fake_map = [
        {'target_faces_in_frame': [{'faces': [], 'location': '/tmp/fake.png'}]}
    ]
    with patch.object(modules.globals, 'source_target_map', fake_map):
        from modules import face_analyser
        face_analyser.default_target_face()
