from types import SimpleNamespace

import pytest

from test_core_map_faces_fallback import _patched_core_import_stubs


@pytest.fixture
def image_run(tmp_path, monkeypatch):
    with _patched_core_import_stubs([]) as core:
        target = tmp_path / 'target.jpg'
        output = tmp_path / 'output.png'
        target.write_bytes(b'original target')
        output.write_bytes(b'existing output')
        core.modules.globals.source_path = 'source.png'
        core.modules.globals.target_path = str(target)
        core.modules.globals.output_path = str(output)
        core.modules.globals.frame_processors = ['face_swapper']
        core.modules.globals.nsfw_filter = False
        core.modules.globals.headless = True
        core.modules.globals.execution_providers = ['CPUExecutionProvider']
        monkeypatch.setattr(core, 'has_image_extension', lambda _: True)
        yield core, target, output


def processor(callback):
    return SimpleNamespace(NAME='test', pre_start=lambda: True, process_image=callback)


def test_failed_processor_preserves_output_and_input(image_run, monkeypatch, capsys):
    core, target, output = image_run
    monkeypatch.setattr(core, 'get_frame_processors_modules', lambda _: [processor(lambda *_: False)])
    assert core.start() is False
    assert target.read_bytes() == b'original target'
    assert output.read_bytes() == b'existing output'
    assert sorted(p.name for p in output.parent.iterdir()) == ['output.png', 'target.jpg']
    assert 'succeed' not in capsys.readouterr().out


def test_pipeline_publishes_only_after_all_processors_succeed(image_run, monkeypatch):
    core, target, output = image_run

    def swap(_source, staging, destination):
        from pathlib import Path
        assert staging == destination
        assert output.read_bytes() == b'existing output'
        Path(destination).write_bytes(b'swapped')
        return True

    def enhance(_source, staging, destination):
        from pathlib import Path
        assert Path(staging).read_bytes() == b'swapped'
        assert output.read_bytes() == b'existing output'
        Path(destination).write_bytes(b'swapped and enhanced')
        return True

    monkeypatch.setattr(core, 'get_frame_processors_modules', lambda _: [processor(swap), processor(enhance)])
    assert core.start() is True
    assert output.read_bytes() == b'swapped and enhanced'
    assert target.read_bytes() == b'original target'
    assert len(list(output.parent.iterdir())) == 2


def test_second_processor_failure_does_not_publish_partial_result(image_run, monkeypatch):
    core, _, output = image_run
    monkeypatch.setattr(core, 'get_frame_processors_modules', lambda _: [processor(lambda *_: True), processor(lambda *_: False)])
    assert core.start() is False
    assert output.read_bytes() == b'existing output'


def test_missing_target_cannot_report_success(image_run, monkeypatch):
    core, target, output = image_run
    target.unlink()
    monkeypatch.setattr(core, 'get_frame_processors_modules', lambda _: [processor(lambda *_: True)])
    assert core.start() is False
    assert output.read_bytes() == b'existing output'
    assert len(list(output.parent.iterdir())) == 1


def test_unexpected_processor_exception_cleans_staging(image_run, monkeypatch):
    core, _, output = image_run

    def fail(*_):
        raise RuntimeError('inference failed')

    monkeypatch.setattr(core, 'get_frame_processors_modules', lambda _: [processor(fail)])
    with pytest.raises(RuntimeError, match='inference failed'):
        core.start()
    assert output.read_bytes() == b'existing output'
    assert len(list(output.parent.iterdir())) == 2


def test_cli_failure_has_nonzero_exit(image_run, monkeypatch):
    core, _, _ = image_run
    monkeypatch.setattr(core, 'parse_args', lambda: None)
    monkeypatch.setattr(core, 'pre_check', lambda: True)
    monkeypatch.setattr(core, 'limit_resources', lambda: None)
    monkeypatch.setattr(core, 'get_frame_processors_modules', lambda _: [])
    monkeypatch.setattr(core, 'start', lambda: False)
    with pytest.raises(SystemExit) as error:
        core.run()
    assert error.value.code == 1


@pytest.mark.parametrize('failure', ['application', 'processor'])
def test_startup_check_failure_has_nonzero_exit(image_run, monkeypatch, failure):
    core, _, _ = image_run
    monkeypatch.setattr(core, 'parse_args', lambda: None)
    monkeypatch.setattr(core, 'pre_check', lambda: failure != 'application')
    monkeypatch.setattr(core, 'get_frame_processors_modules', lambda _: [
        SimpleNamespace(pre_check=lambda: failure != 'processor')
    ])
    with pytest.raises(SystemExit) as error:
        core.run()
    assert error.value.code == 1
