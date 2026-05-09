import ast
from pathlib import Path


def _core_ast():
    core_path = Path(__file__).resolve().parents[1] / 'modules' / 'core.py'
    return ast.parse(core_path.read_text())


def _add_argument_calls():
    for node in ast.walk(_core_ast()):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument':
            yield node


def _literal_args(call):
    values = []
    for arg in call.args:
        if isinstance(arg, ast.Constant):
            values.append(arg.value)
    return values


def _keyword_value(call, name):
    for kw in call.keywords:
        if kw.arg == name and isinstance(kw.value, ast.Constant):
            return kw.value.value
    return None


def test_no_keep_audio_cli_flag_disables_audio_restore():
    no_keep_audio = [call for call in _add_argument_calls() if '--no-keep-audio' in _literal_args(call)]

    assert no_keep_audio, 'headless CLI should expose --no-keep-audio'
    assert _keyword_value(no_keep_audio[0], 'dest') == 'keep_audio'
    assert _keyword_value(no_keep_audio[0], 'action') == 'store_false'


def _set_defaults_calls():
    for node in ast.walk(_core_ast()):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'set_defaults':
            yield node


def test_keep_audio_default_stays_enabled():
    keep_audio = [call for call in _add_argument_calls() if '--keep-audio' in _literal_args(call)]
    defaults = list(_set_defaults_calls())

    assert keep_audio, 'existing --keep-audio flag should remain available'
    assert _keyword_value(keep_audio[0], 'dest') == 'keep_audio'
    assert any(_keyword_value(call, 'keep_audio') is True for call in defaults)
