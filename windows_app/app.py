from __future__ import annotations

from pathlib import Path

from windows_app import app_base as base
from windows_app.live_webcam import LiveWebcamMixin, load_settings, save_settings
from windows_app.main_window_ui import MainWindowUiMixin
from windows_app.output_tasks import OutputTasksMixin
from windows_app.processing_options import ProcessingOptionsMixin

# Keep app_base settings functions aligned for helpers that call base.save_settings().
base.load_settings = load_settings
base.save_settings = save_settings

# Re-export common application symbols for lightweight compatibility with callers.
AppSettings = base.AppSettings
ApiClient = base.ApiClient
PollWorker = base.PollWorker
LiveWorker = base.LiveWorker
DEFAULT_DRIVE_ROOT = base.DEFAULT_DRIVE_ROOT
APP_STATE = base.APP_STATE
PHOTO_EXTENSIONS = base.PHOTO_EXTENSIONS
VIDEO_EXTENSIONS = base.VIDEO_EXTENSIONS
QApplication = base.QApplication


class MainWindow(LiveWebcamMixin, ProcessingOptionsMixin, MainWindowUiMixin, OutputTasksMixin, base.MainWindow):
    """Canonical Windows remote controller window.

    The mixins replace the former runtime patch stack with normal class
    composition while preserving the same method resolution order as the old
    import chain: async outputs -> UI -> processing options -> live webcam.
    """


# Ensure workers/helpers that instantiate base.MainWindow-compatible behavior
# see the canonical class when they intentionally reference this module.
base.MainWindow = MainWindow


def main() -> int:
    app = base.QApplication([])

    qss_path = Path(__file__).parent / "dark_theme.qss"
    if qss_path.exists():
        app.setStyleSheet(qss_path.read_text(encoding="utf-8"))

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
