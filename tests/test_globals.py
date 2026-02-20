"""Tests that globals required by face masking are defined and have the right type."""
import pytest


def test_globals_mask_size_attributes():
    import modules.globals
    assert hasattr(modules.globals, 'mouth_mask_size')
    assert hasattr(modules.globals, 'eyes_mask_size')
    assert hasattr(modules.globals, 'eyebrows_mask_size')
    assert isinstance(modules.globals.mouth_mask_size, float)
    assert isinstance(modules.globals.eyes_mask_size, float)
    assert isinstance(modules.globals.eyebrows_mask_size, float)


def test_globals_fps_cap_is_float():
    import modules.globals
    assert isinstance(modules.globals.FPS_CAP, float)
