"""Local minimal model preparation regressions; no external requests."""

import os
from pathlib import Path
from unittest.mock import Mock

# Disable the optional dependency's import-time network version check.
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import pytest

from modules import face_analyser, model_downloader
from modules.processors.frame import _onnx_enhancer


def test_pack_is_local_minimal_and_idempotent(tmp_path, monkeypatch):
    models_dir = tmp_path / "installation" / "models"
    unrelated = tmp_path / "home" / ".insightface" / "models" / "old.onnx"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_bytes(b"existing user model")
    monkeypatch.setattr(model_downloader, "MODELS_DIR", str(models_dir))
    monkeypatch.setattr(
        model_downloader.os.path,
        "expanduser",
        Mock(
            side_effect=AssertionError(
                "model setup must not resolve the home directory"
            )
        ),
    )
    downloads = []

    def download(name, url, target, size):
        assert url == model_downloader.resolve_url(name)
        assert size == model_downloader.expected_size(name)
        downloads.append((name, Path(target)))
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        Path(target).write_bytes(b"downloaded model")
        return True

    monkeypatch.setattr(model_downloader, "_download", download)

    assert model_downloader.ensure_insightface_pack()
    assert {path.name for _, path in downloads} == {
        "det_10g.onnx",
        "w600k_r50.onnx",
        "2d106det.onnx",
    }
    assert all(path.parent == models_dir / "buffalo_l" for _, path in downloads)
    assert model_downloader.ensure_insightface_pack()
    assert len(downloads) == 3
    assert unrelated.read_bytes() == b"existing user model"

def test_unknown_pack_does_not_attempt_download(monkeypatch):
    download = Mock(side_effect=AssertionError("unknown pack must not download"))
    monkeypatch.setattr(model_downloader, "_download", download)

    assert not model_downloader.ensure_insightface_pack("unknown-pack")
    download.assert_not_called()

def test_analyser_uses_the_prepared_local_pack(tmp_path, monkeypatch):
    root = tmp_path / "installation"
    pack = root / "models" / "buffalo_l"
    pack.mkdir(parents=True)
    for filename in model_downloader.INSIGHTFACE_MODEL_FILES.values():
        (pack / filename).write_bytes(b"local model")
    monkeypatch.setattr(model_downloader, "MODELS_DIR", str(root / "models"))
    monkeypatch.setattr(model_downloader, "INSIGHTFACE_ROOT", str(root))
    monkeypatch.setattr(face_analyser, "FACE_ANALYSER", None)
    monkeypatch.setattr(face_analyser.modules.globals, "det_size", 320)
    monkeypatch.setattr(
        _onnx_enhancer,
        "build_provider_config",
        lambda **_kwargs: ["CPUExecutionProvider"],
    )
    monkeypatch.setattr(face_analyser, "_optimize_det_model", Mock())
    download = Mock(side_effect=AssertionError("prepared pack must not download"))
    monkeypatch.setattr(model_downloader, "_download", download)
    analyser = Mock()
    factory = Mock(return_value=analyser)
    monkeypatch.setattr(face_analyser.insightface.app, "FaceAnalysis", factory)

    assert face_analyser.get_face_analyser() is analyser
    assert face_analyser.get_face_analyser() is analyser
    factory.assert_called_once_with(
        name="buffalo_l",
        root=str(root),
        providers=["CPUExecutionProvider"],
        allowed_modules=list(model_downloader.INSIGHTFACE_MODEL_FILES),
    )
    analyser.prepare.assert_called_once_with(ctx_id=0, det_size=(320, 320))
    download.assert_not_called()

def test_incomplete_local_pack_stops_before_insightface_fallback(monkeypatch):
    monkeypatch.setattr(face_analyser, "FACE_ANALYSER", None)
    monkeypatch.setattr(
        model_downloader, "ensure_insightface_pack", lambda _name: False
    )
    factory = Mock(side_effect=AssertionError("unprepared pack must not be loaded"))
    monkeypatch.setattr(face_analyser.insightface.app, "FaceAnalysis", factory)

    with pytest.raises(RuntimeError, match="Could not prepare local InsightFace"):
        face_analyser.get_face_analyser()
    assert face_analyser.FACE_ANALYSER is None
    factory.assert_not_called()
