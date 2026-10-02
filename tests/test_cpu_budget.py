"""Trainer CPU threads: a share of the container's CPU quota or affinity."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from slime.utils.cpu_budget import available_cpus, threads_per_rank


class CpuBudgetTests(unittest.TestCase):
    def test_quota_caps_affinity_and_is_shared_by_ranks(self):
        with tempfile.TemporaryDirectory() as folder, patch('os.sched_getaffinity', return_value=set(range(240))):
            v2 = Path(folder, 'cpu.max')
            missing = str(Path(folder, 'none'))
            v2.write_text('5000000 100000\n')  # the remote test container: 50 CPUs
            self.assertEqual(available_cpus(str(v2), missing, missing), 50)
            self.assertEqual(threads_per_rank(2, cpus=50), 25)
            v2.write_text('max 100000\n')  # no quota: the affinity
            self.assertEqual(available_cpus(str(v2), missing, missing), 240)
            Path(folder, 'quota').write_text('3200000\n')  # cgroup v1
            Path(folder, 'period').write_text('100000\n')
            self.assertEqual(available_cpus(missing, str(Path(folder, 'quota')), str(Path(folder, 'period'))), 32)
            Path(folder, 'quota').write_text('-1\n')  # v1 without a quota
            self.assertEqual(available_cpus(missing, str(Path(folder, 'quota')), str(Path(folder, 'period'))), 240)
        self.assertEqual(threads_per_rank(6, cpus=4), 1)


if __name__ == '__main__':
    unittest.main()
