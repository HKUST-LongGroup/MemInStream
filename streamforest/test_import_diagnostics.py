import importlib
from pathlib import Path
import sys
import tempfile
import unittest

from check_model_imports import import_with_original_error


class ImportDiagnosticsTests(unittest.TestCase):
    def test_hidden_error_and_trace_restoration(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / 'fixture_model_import'
            package.mkdir()
            source = package / '__init__.py'
            source.write_text('try:\n    import nonexistent_streamforest_dependency_test\n'
                              'except Exception:\n    pass\n'
                              'raise ImportError("secondary missing model class")\n')
            sys.path.insert(0, directory)
            previous = sys.gettrace()
            try:
                with self.assertRaises(RuntimeError) as result:
                    import_with_original_error('fixture_model_import', source)
                # The final propagated exception is also observed by the trace;
                # the swallowed dependency error must take precedence.
                self.assertIsInstance(result.exception.__cause__, ModuleNotFoundError)
                self.assertIn('nonexistent_streamforest_dependency_test', str(result.exception))
                self.assertIs(sys.gettrace(), previous)
            finally:
                sys.path.remove(directory)
                sys.modules.pop('fixture_model_import', None)

    def test_successful_import(self):
        previous = sys.gettrace()
        self.assertIs(import_with_original_error('json', '/nonexistent/catch.py'),
                      importlib.import_module('json'))
        self.assertIs(sys.gettrace(), previous)
