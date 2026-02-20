"""Tests for modules/processors/frame/core.py."""
import pytest
from unittest.mock import MagicMock, patch


def test_load_frame_processor_raises_for_missing_method():
    """load_frame_processor_module raises ImportError for a module missing a required method."""
    from modules.processors.frame.core import load_frame_processor_module

    # Create a fake module that is missing 'process_video'
    fake_module = MagicMock(spec=["pre_check", "pre_start", "process_frame", "process_image"])

    with patch("importlib.import_module", return_value=fake_module):
        with pytest.raises(ImportError, match="missing method"):
            load_frame_processor_module("fake_processor")


def test_load_frame_processor_raises_on_import_error():
    """load_frame_processor_module raises ImportError when the module does not exist."""
    from modules.processors.frame.core import load_frame_processor_module

    with pytest.raises(ImportError):
        load_frame_processor_module("nonexistent_processor_xyz")
