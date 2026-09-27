import tempfile
import unittest
from pathlib import Path
from slime_plugins.chimera_mixrl.budget import RunBudget


class BudgetTests(unittest.TestCase):
    def test_reserve_and_measured_whole_update_latency(self):
        now = [0.]
        budget = RunBudget(10800, 1200, 300, clock=lambda: now[0])
        now[0] = 400  # initialization is included in rental clock
        budget.begin_update()
        now[0] = 800
        self.assertFalse(budget.finish_update())
        self.assertEqual(budget.estimate, 400)
        now[0] = 9200
        self.assertTrue(budget.should_stop())

    def test_disabled_and_stop_file(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'STOP'
            budget = RunBudget(stop_file=str(path))
            self.assertFalse(budget.should_stop())
            path.touch()
            self.assertTrue(budget.should_stop())

    def test_invalid_budget(self):
        with self.assertRaises(ValueError):
            RunBudget(100, 100)
