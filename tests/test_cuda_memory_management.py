import builtins
import importlib
import sys
import types
import unittest
from unittest.mock import Mock, patch


def _frame_core_import_stubs():
    return {
        'cv2': types.SimpleNamespace(
            IMREAD_COLOR=1,
            imread=lambda *_args, **_kwargs: None,
            imdecode=lambda *_args, **_kwargs: None,
            imencode=lambda *_args, **_kwargs: (
                True,
                types.SimpleNamespace(tofile=lambda *_a, **_k: None),
            ),
        ),
        'numpy': types.SimpleNamespace(uint8=object),
        'tqdm': types.SimpleNamespace(tqdm=lambda *args, **_kwargs: args[0]),
        'modules.face_analyser': types.SimpleNamespace(
            get_one_face=lambda *_args, **_kwargs: None,
        ),
    }


def _unload_modules():
    for module_name in tuple(sys.modules):
        if module_name == 'modules' or module_name.startswith('modules.'):
            sys.modules.pop(module_name)


class FakeCuda:
    def __init__(self, allocated, reserved, is_available=True):
        self.allocated = list(allocated)
        self.reserved = list(reserved)
        self.is_available_value = is_available
        self.empty_cache_calls = 0

    def is_available(self):
        return self.is_available_value

    def memory_allocated(self):
        return self.allocated.pop(0)

    def memory_reserved(self):
        return self.reserved.pop(0)

    def empty_cache(self):
        self.empty_cache_calls += 1


class CudaMemoryManagementTests(unittest.TestCase):
    def setUp(self):
        self.existing_modules = {
            name: module
            for name, module in sys.modules.items()
            if name == 'modules' or name.startswith('modules.')
        }
        _unload_modules()
        self.import_stubs = patch.dict(sys.modules, _frame_core_import_stubs())
        self.import_stubs.start()
        self.frame_core = importlib.import_module('modules.processors.frame.core')
        self.execution_providers = self.frame_core.modules.globals.execution_providers

    def tearDown(self):
        self.frame_core.modules.globals.execution_providers = self.execution_providers
        _unload_modules()
        self.import_stubs.stop()
        sys.modules.update(self.existing_modules)

    def _use_cuda_provider(self):
        self.frame_core.modules.globals.execution_providers = ['CUDAExecutionProvider']

    def test_does_not_import_torch_without_cuda_execution_provider(self):
        self.frame_core.modules.globals.execution_providers = ['CPUExecutionProvider']

        with patch('builtins.__import__', side_effect=AssertionError('torch import')):
            self.assertIsNone(self.frame_core._get_cuda_torch())

    def test_returns_none_when_torch_import_fails(self):
        self._use_cuda_provider()
        original_import = builtins.__import__

        def import_without_torch(name, *args, **kwargs):
            if name == 'torch':
                raise ImportError('torch is unavailable')
            return original_import(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=import_without_torch):
            self.assertIsNone(self.frame_core._get_cuda_torch())

    def test_returns_torch_only_when_cuda_is_available(self):
        fake_cuda = FakeCuda([], [])
        fake_torch = types.SimpleNamespace(cuda=fake_cuda)
        self._use_cuda_provider()

        with patch.dict(sys.modules, {'torch': fake_torch}):
            self.assertIs(self.frame_core._get_cuda_torch(), fake_torch)

    def test_skips_torch_without_an_available_cuda_device(self):
        fake_cuda = FakeCuda([], [], is_available=False)
        fake_torch = types.SimpleNamespace(cuda=fake_cuda)
        self._use_cuda_provider()

        with patch.dict(sys.modules, {'torch': fake_torch}):
            self.assertIsNone(self.frame_core._get_cuda_torch())

    def test_returns_none_when_cuda_probe_fails(self):
        fake_cuda = types.SimpleNamespace(
            is_available=Mock(side_effect=RuntimeError('CUDA probing failed')),
        )
        self._use_cuda_provider()

        with patch.dict(sys.modules, {'torch': types.SimpleNamespace(cuda=fake_cuda)}):
            self.assertIsNone(self.frame_core._get_cuda_torch())

    def test_returns_none_when_memory_stats_fail(self):
        fake_cuda = types.SimpleNamespace(
            memory_allocated=Mock(side_effect=RuntimeError('CUDA stats failed')),
            memory_reserved=Mock(),
        )

        self.assertIsNone(
            self.frame_core._get_cuda_memory_stats(types.SimpleNamespace(cuda=fake_cuda)),
        )

    def test_reports_allocated_and_reserved_memory_after_cache_cleanup(self):
        mebibyte = 1024 * 1024
        fake_cuda = FakeCuda([6 * mebibyte, 4 * mebibyte], [10 * mebibyte, 5 * mebibyte])
        fake_torch = types.SimpleNamespace(cuda=fake_cuda)

        with patch.object(self.frame_core.gc, 'collect') as collect, patch.object(
            self.frame_core, '_report_cuda_cache_cleanup',
        ) as report:
            self.frame_core._clear_cuda_cache(fake_torch, 50)

        collect.assert_not_called()
        self.assertEqual(fake_cuda.empty_cache_calls, 1)
        report.assert_called_once_with(
            'CUDA cache cleanup at frame 50: '
            'allocated 6.0 MiB -> 4.0 MiB, reserved 10.0 MiB -> 5.0 MiB',
        )

    def test_runs_full_gc_at_the_lower_frequency_interval(self):
        mebibyte = 1024 * 1024
        fake_cuda = FakeCuda([6 * mebibyte, 4 * mebibyte], [10 * mebibyte, 5 * mebibyte])
        fake_torch = types.SimpleNamespace(cuda=fake_cuda)

        with patch.object(self.frame_core.gc, 'collect') as collect, patch.object(
            self.frame_core, '_report_cuda_cache_cleanup',
        ):
            self.frame_core._clear_cuda_cache(
                fake_torch,
                self.frame_core._CUDA_GC_COLLECT_INTERVAL,
            )

        collect.assert_called_once_with()

    def test_ignores_empty_cache_runtime_errors(self):
        fake_cuda = types.SimpleNamespace(
            memory_allocated=Mock(return_value=0),
            memory_reserved=Mock(return_value=0),
            empty_cache=Mock(side_effect=RuntimeError('CUDA cache unavailable')),
        )

        with patch.object(self.frame_core, '_report_cuda_cache_cleanup') as report:
            self.frame_core._clear_cuda_cache(types.SimpleNamespace(cuda=fake_cuda), 50)

        report.assert_not_called()


if __name__ == '__main__':
    unittest.main()
