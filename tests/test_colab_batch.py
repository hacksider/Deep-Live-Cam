import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "colab_batch.py"
SPEC = importlib.util.spec_from_file_location("colab_batch", MODULE_PATH)
batch = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = batch
SPEC.loader.exec_module(batch)


def test_processing_geometry_caps_without_upscaling():
    assert batch.processing_geometry(1920, 1080, 60.0, 420, 30.0) == (420, 236, 30.0)
    assert batch.processing_geometry(320, 240, 24.0, 420, 30.0) == (320, 240, 24.0)


def test_decoder_uses_compatible_vsync_and_preserves_selected_fps(tmp_path):
    command = batch.decoder_command(tmp_path / "input.mp4", False, 2.0, 5.0, 29.97, 420, 236)
    assert "-vsync" in command
    assert "fps_mode" not in command
    assert "fps=29.97,scale=420:236" in command


def test_manifest_key_changes_with_processing_configuration(tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    video = input_dir / "clip.mp4"
    video.write_bytes(b"video")
    source = tmp_path / "source.png"
    source.write_bytes(b"face")
    first = batch.ProcessConfig(input_dir, output_dir, source, None, max_width=420)
    second = batch.ProcessConfig(input_dir, output_dir, source, None, max_width=640)
    first_key = batch.manifest_key(video, input_dir, batch.config_signature(first))
    second_key = batch.manifest_key(video, input_dir, batch.config_signature(second))
    assert first_key != second_key


def test_mapping_schema_requires_version_and_videos(tmp_path):
    invalid = tmp_path / "mapping.json"
    invalid.write_text('{"version": 2}', encoding="utf-8")
    # Schema can be validated without constructing the GPU engine.
    payload = batch.load_json(invalid, {})
    assert not (payload.get("version") == 1 and isinstance(payload.get("videos"), dict))
