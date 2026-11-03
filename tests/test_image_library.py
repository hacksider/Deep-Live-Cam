import os
import shutil
import tempfile
import pytest
import numpy as np
import cv2

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication
import modules.globals
import modules.paths as paths
from modules.ui import ImageLibraryDialog, MainWindow


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def temp_library_env(monkeypatch):
    tmpdir = tempfile.mkdtemp()
    sources_dir = os.path.join(tmpdir, "sources")
    bg_dir = os.path.join(tmpdir, "backgrounds")
    monkeypatch.setattr(paths, "LIBRARY_DIR", tmpdir)
    monkeypatch.setattr(paths, "LIBRARY_SOURCES_DIR", sources_dir)
    monkeypatch.setattr(paths, "LIBRARY_BACKGROUNDS_DIR", bg_dir)

    paths.ensure_library_dirs()

    yield tmpdir, sources_dir, bg_dir

    shutil.rmtree(tmpdir, ignore_errors=True)


def _create_dummy_image(filepath: str, width: int = 100, height: int = 100):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = (120, 150, 180)
    cv2.imwrite(filepath, img)


def test_ensure_library_dirs(temp_library_env):
    tmpdir, sources_dir, bg_dir = temp_library_env
    assert os.path.isdir(sources_dir)
    assert os.path.isdir(bg_dir)


def test_image_library_dialog_display_and_selection(qapp, temp_library_env):
    tmpdir, sources_dir, bg_dir = temp_library_env
    src_img = os.path.join(sources_dir, "face1.png")
    bg_img = os.path.join(bg_dir, "bg1.png")
    _create_dummy_image(src_img)
    _create_dummy_image(bg_img)

    selected_source = None
    selected_bg = None

    def on_src(p):
        nonlocal selected_source
        selected_source = p

    def on_bg(p):
        nonlocal selected_bg
        selected_bg = p

    dlg = ImageLibraryDialog(
        initial_category=ImageLibraryDialog.CATEGORY_SOURCES,
        on_source_selected=on_src,
        on_background_selected=on_bg,
    )

    assert len(dlg._items) == 1
    assert dlg._items[0].image_path == src_img
    assert not dlg.btn_use_source.isEnabled()

    # Click item
    dlg._on_item_clicked(src_img)
    assert dlg.btn_use_source.isEnabled()
    assert dlg.btn_use_bg.isEnabled()

    # Apply as source
    dlg._on_apply_as_source()
    assert selected_source == src_img

    # Switch category to backgrounds
    dlg._set_category(ImageLibraryDialog.CATEGORY_BACKGROUNDS)
    assert len(dlg._items) == 1
    assert dlg._items[0].image_path == bg_img

    dlg._on_item_clicked(bg_img)
    dlg._on_apply_as_bg()
    assert selected_bg == bg_img


def test_main_window_set_paths(qapp, temp_library_env):
    tmpdir, sources_dir, bg_dir = temp_library_env
    src_img = os.path.join(sources_dir, "test_face.png")
    bg_img = os.path.join(bg_dir, "test_bg.png")
    _create_dummy_image(src_img)
    _create_dummy_image(bg_img)

    win = MainWindow(lambda: None, lambda: None)

    # Test setting source path via helper
    win._set_source_path(src_img)
    assert modules.globals.source_path == src_img
    assert win.source_label.pixmap() is not None

    # Test setting background path via helper
    win._set_background_path(bg_img)
    assert modules.globals.background_path == bg_img
    assert win.bg_label.text() == "test_bg.png"

    # Test clearing background
    win._on_clear_background()
    assert modules.globals.background_path is None
    assert "No background selected" in win.bg_label.text()
