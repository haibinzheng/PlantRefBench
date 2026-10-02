import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load(relative):
    spec = importlib.util.spec_from_file_location('test_target', ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class PackageTests(unittest.TestCase):
    def test_archive_syntax_and_local_imports(self):
        self.assertGreater(load('scripts/check_package.py').check()['python_sources_parsed'], 20)

    def test_example(self):
        result = load('examples/validate_example.py').validate()
        self.assertEqual((result['reference_cells'], result['target_cells'], result['orthogroup_features']), (10, 4, 4))

    def test_target_label_leak_is_rejected(self):
        original = (ROOT / 'examples/synthetic_orthogroup_cells.csv').read_text(encoding='utf-8')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'leaky.csv'
            path.write_text(original.replace('target,,', 'target,cortex,'), encoding='utf-8')
            with self.assertRaises(AssertionError):
                load('examples/validate_example.py').validate(path)

    def test_duplicate_cell_is_rejected(self):
        original = (ROOT / 'examples/synthetic_orthogroup_cells.csv').read_text(encoding='utf-8')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'duplicate.csv'
            path.write_text(original.replace('toy_at_002', 'toy_at_001'), encoding='utf-8')
            with self.assertRaises(AssertionError):
                load('examples/validate_example.py').validate(path)

if __name__ == '__main__':
    unittest.main()

