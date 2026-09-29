"""The MixRL task file: schema, domains, eval sizing rules, preview, and grading readiness."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from slime_plugins.chimera_mixrl import tasks

REPO = Path(__file__).resolve().parents[1]
JUDGE_FREE = {'gsm8k_train', 'mcqa', 'nemotron_if', 'calendar', 'apps'}


def ready_health(spec, **overrides):
    health = {'judge': {'model': 'glimmer', 'ready': True}, 'task_errors': {},
              'task_judge': {n: t['about']['judge'] for n, t in spec['tasks'].items()}}
    health.update(overrides)
    return health


class TaskFileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'tasks.json'
        self.default = json.loads(tasks.DEFAULT_PATH.read_text())

    def write(self, change):
        spec = copy.deepcopy(self.default)
        change(spec)
        self.path.write_text(json.dumps(spec))
        return self.path

    def only(self, *names, prompts=None):
        def change(spec):
            for name, task in spec['tasks'].items():
                task['enabled'] = name in names
                if prompts and name in names:
                    task['prompts_per_step'] = prompts
        return change

    def test_default_file_is_the_sixteen_task_production_mix(self):
        spec = tasks.load()
        self.assertEqual((len(spec['tasks']), len(spec['domains'])), (16, 9))
        resolved = tasks.resolved(spec)
        self.assertEqual(resolved['rollout_batch_size'], 512)
        self.assertEqual({n for n, t in spec['tasks'].items() if t['about']['judge'] == 'none'}, JUDGE_FREE)
        # "all": every validation prompt; 506 in total (v5), 52-57 per domain.
        self.assertEqual(sum(resolved['eval_quotas'].values()), 506)
        self.assertEqual(resolved['eval_quotas']['mcqa'], 19)
        self.assertLessEqual(tasks.eval_cost(spec, 8)['steps'], 1)

    def test_gsm8k_only_gets_the_whole_math_eval(self):
        spec = tasks.load(self.write(self.only('gsm8k_train', prompts=64)))
        resolved = tasks.resolved(spec)
        self.assertEqual(resolved['quotas'], {'gsm8k_train': 64})
        self.assertEqual(resolved['eval_quotas'], {'gsm8k_train': 29})
        self.assertEqual(resolved['rollout_batch_size'], 64)
        self.assertAlmostEqual(tasks.eval_cost(spec, 8)['steps'], 29 * 4 / (64 * 8))

    def test_eval_is_chosen_per_task_like_training(self):
        def choose(spec):
            self.only('mcqa', 'openqa', 'science')(spec)
            spec['tasks']['mcqa']['eval_prompts'] = 10
            spec['tasks']['science']['eval_prompts'] = 50
        spec = tasks.load(self.write(choose))
        # A number up to the pool; "all" (the default) is the whole pool; more than the pool is capped.
        self.assertEqual(tasks.eval_counts(spec), {'mcqa': 10, 'openqa': 19, 'science': 19})
        self.assertEqual(tasks.notes(spec), ['note: science asks for 50 eval prompts; its validation pool has 19'])
        for bad in (0, 'some', 2.5):
            with self.subTest(bad), self.assertRaisesRegex(ValueError, 'mcqa: eval_prompts must be "all" or a positive integer'):
                tasks.load(self.write(lambda s: s['tasks']['mcqa'].update(eval_prompts=bad)))

    def test_eval_cost_is_reported_not_enforced(self):
        spec = tasks.load(self.write(self.only('gsm8k_train', prompts=2)))
        self.assertAlmostEqual(tasks.eval_cost(spec, 8)['steps'], 29 * 4 / (2 * 8))
        self.assertIn('up to 7.25 training steps of tokens', tasks.table(spec, 8))

    def test_schema_errors_name_the_task_and_field(self):
        cases = {
            'gsm8k_train: fields must be exactly': lambda s: s['tasks']['gsm8k_train'].update(weight=1),
            'enabled must be true or false': lambda s: s['tasks']['mcqa'].update(enabled='yes'),
            'prompts_per_step must be a positive integer': lambda s: s['tasks']['mcqa'].update(prompts_per_step=0),
            'prompts_per_step 700 exceeds train_pool 623': lambda s: s['tasks']['cascade_plans'].update(prompts_per_step=700),
            'about must have exactly': lambda s: s['tasks']['mcqa']['about'].pop('answer_format'),
            'about.requires must be a list of strings': lambda s: s['tasks']['mcqa']['about'].update(requires='regex'),
            'about.domain chemistry is not listed': lambda s: s['tasks']['mcqa']['about'].update(domain='chemistry'),
            'about.judge must be one of': lambda s: s['tasks']['mcqa']['about'].update(judge='maybe'),
            'domains without tasks: chemistry': lambda s: s['domains'].update(chemistry={'summary': 'x'}),
            'domain math needs exactly a summary': lambda s: s['domains']['math'].update(eval_prompts=5),
            'eval must be exactly': lambda s: s['eval'].update(samples_per_prompt=0),
            'no task is enabled': lambda s: [t.update(enabled=False) for t in s['tasks'].values()],
        }
        for message, change in cases.items():
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                tasks.load(self.write(change))

    def test_training_refuses_tasks_the_reward_service_cannot_grade(self):
        spec = tasks.load()
        tasks.check_scorer(spec, ready_health(spec))
        down = ready_health(spec, judge={'model': 'glimmer', 'ready': False})
        with self.assertRaisesRegex(ValueError, "nemotron_math: needs the judge \\(always\\), but judge 'glimmer' is not reachable"):
            tasks.check_scorer(spec, down)
        free = tasks.load(self.write(self.only(*JUDGE_FREE)))
        tasks.check_scorer(free, down)
        with self.assertRaisesRegex(ValueError, "nemotron_if: grading check failed: ModuleNotFoundError: No module named 'ifbench'"):
            tasks.check_scorer(free, dict(down, task_errors={'nemotron_if': "ModuleNotFoundError: No module named 'ifbench'"}))
        # Errors in disabled tasks do not block training.
        tasks.check_scorer(free, dict(down, task_errors={'hotpot_train': 'broken'}))
        with self.assertRaisesRegex(ValueError, 'hotpot_train: task file says judge=always'):
            tasks.check_scorer(free, dict(down, task_judge=dict(down['task_judge'], hotpot_train='none')))
        with self.assertRaisesRegex(ValueError, 'does not report judge status'):
            tasks.check_scorer(free, {'task_judge': down['task_judge'], 'judge': None})

    def test_preview_cli(self):
        result = subprocess.run([sys.executable, '-m', 'slime_plugins.chimera_mixrl.tasks',
                                 str(self.write(self.only(*JUDGE_FREE, prompts=64))), '--samples-per-prompt', '8'],
                                cwd=REPO, capture_output=True, text=True, check=True)
        self.assertIn('5 of 16 tasks enabled; 320 prompts per step x 8 responses = 2560 samples', result.stdout)
        self.assertIn('eval: 185 prompts x 4 samples = 740 per eval; up to 0.29 training steps of tokens\n', result.stdout)
        self.assertIn('judge: not needed', result.stdout)
        self.assertRegex(result.stdout, r'math\s+1/2\s+64\s+29\s+29\s+no')
        self.assertRegex(result.stdout, r'gsm8k_train\s+math\s+yes\s+64\s+4984\s+29\s+29\s+8192')
        self.assertRegex(result.stdout, r'grounding\s+0/1\s+0\s+-\s+-\s+-')
        bad = subprocess.run([sys.executable, '-m', 'slime_plugins.chimera_mixrl.tasks',
                              str(self.write(lambda s: s['tasks']['mcqa'].update(enabled=1)))],
                             cwd=REPO, capture_output=True, text=True)
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn('mcqa: enabled must be true or false', bad.stderr)


if __name__ == '__main__':
    unittest.main()
