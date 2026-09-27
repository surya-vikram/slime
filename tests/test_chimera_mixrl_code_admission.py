import tempfile
import unittest
from pathlib import Path

from slime_plugins.chimera_mixrl.code_admission import code_exclusions
from slime_plugins.chimera_mixrl.core import digest, write_json


class CodeAdmissionTests(unittest.TestCase):
    def test_missing_incomplete_changed_and_valid_audit(self):
        row = dict(id='fixture', task='apps', messages=[{'role': 'user', 'content': 'Print the sum'}])
        with self.assertRaises(ValueError):
            code_exclusions([row], None)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_json(root / 'summary.json', {'total': 1})
            record = {'row_hash': digest(row), 'passed': True,
                      'negative': {'status': 'valid', 'passed': False}}
            write_json(root / 'fixture.json', record)
            self.assertEqual(code_exclusions([row], root)[0], {})
            with self.assertRaises(ValueError):
                code_exclusions([dict(row, messages=[])], root)
            write_json(root / 'summary.json', {'total': 0})
            with self.assertRaises(ValueError):
                code_exclusions([row], root)

    def test_reference_failure_and_ambiguous_output_are_excluded(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_json(root / 'summary.json', {'total': 1})
            for passed, content in ((False, 'Print the sum'), (True, 'Output any valid solution')):
                row = dict(id='fixture', task='apps', messages=[{'role': 'user', 'content': content}])
                write_json(root / 'fixture.json', dict(row_hash=digest(row), passed=passed,
                            negative={'status': 'valid', 'passed': False}))
                excluded, _ = code_exclusions([row], root)
                self.assertIn('fixture', excluded)


if __name__ == '__main__':
    unittest.main()
