"""Tests for modules/utilities.py."""
import pytest


def test_has_image_extension_png():
    from modules.utilities import has_image_extension
    assert has_image_extension("photo.png") is True


def test_has_image_extension_jpg():
    from modules.utilities import has_image_extension
    assert has_image_extension("photo.jpg") is True


def test_has_image_extension_jpeg():
    from modules.utilities import has_image_extension
    assert has_image_extension("photo.jpeg") is True


def test_has_image_extension_gif():
    from modules.utilities import has_image_extension
    assert has_image_extension("photo.gif") is True


def test_has_image_extension_bmp():
    from modules.utilities import has_image_extension
    assert has_image_extension("photo.bmp") is True


def test_has_image_extension_uppercase():
    from modules.utilities import has_image_extension
    assert has_image_extension("photo.PNG") is True


def test_has_image_extension_false_for_video():
    from modules.utilities import has_image_extension
    assert has_image_extension("video.mp4") is False


def test_has_image_extension_false_no_extension():
    from modules.utilities import has_image_extension
    assert has_image_extension("noextension") is False


def test_has_image_extension_rejects_bare_png_without_dot():
    """'somepng' should not match — requires the dot prefix."""
    from modules.utilities import has_image_extension
    # "somepng" ends with "png" but not ".png"
    assert has_image_extension("somepng") is False
