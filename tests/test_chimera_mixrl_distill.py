"""Distillation config (slime_plugins/chimera_mixrl/distill.py, mixrl/distill.json): paths, tasks, teachers, batch,
server plan, preview and eval summary. Standard library only."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from slime_plugins.chimera_mixrl import distill

REPO = Path(__file__).resolve().parents[1]
CONFIG = {'architectures': ['ChimeraForCausalLM'], 'hidden_size': 64, 'num_hidden_layers': 2, 'vocab_size': 100,
          'transformers_version': '4.57.0'}


def checkpoint(path, config=CONFIG, template='{{ messages }}'):
    path.mkdir(parents=True, exist_ok=True)
    (path / 'config.json').write_text(json.dumps(config))
    (path / 'tokenizer.json').write_text('{"model": "fixture"}')
    (path / 'chat_template.jinja').write_text(template)
    (path / 'model.safetensors').write_bytes(b'weights')


def splits(path, counts):
    """counts: {task: (rl_train rows, rl_val rows)} -> a splits folder like the HF release's."""
    path.mkdir(parents=True, exist_ok=True)
    for i, split in enumerate(('rl_train', 'rl_val')):
        with open(path / f'{split}.jsonl', 'w') as stream:
            for task, sizes in counts.items():
                for j in range(sizes[i]):
                    stream.write(json.dumps({'id': f'{task}-{split}-{j}', 'family_id': f'{task}-{split}-{j}', 'task': task,
                                             'domain': 'fixture', 'binary': True,
                                             'messages': [{'role': 'user', 'content': 'q' * 40}]}) + '\n')
    (path / 'main_test.jsonl').write_text('')


def make_config(root, tasks=None, teachers=None, counts=None):
    """A complete fixture: student, splits, two teachers and a config; returns the config path."""
    checkpoint(root / 'zoro3' / 'hf')
    (root / 'zoro3' / 'mcore').mkdir(parents=True, exist_ok=True)
    (root / 'zoro3' / 'mcore' / 'latest_checkpointed_iteration.txt').write_text('100')
    splits(root / 'data' / 'splits', counts or {'gsm8k_train': (900, 29), 'mcqa': (1800, 19),
                                                 'nvidia_multichallenge': (300, 29)})
    for name in ('four_tasks', 'multiturn'):
        checkpoint(root / 'teachers' / name)
    config = {'student': {'hf': str(root / 'zoro3' / 'hf'), 'mcore': str(root / 'zoro3' / 'mcore')},
              'splits': str(root / 'data' / 'splits'),
              'teachers': teachers or {'four_tasks': str(root / 'teachers' / 'four_tasks'),
                                       'multiturn': str(root / 'teachers' / 'multiturn')},
              'tasks': tasks or {'gsm8k_train': {'teacher': 'four_tasks', 'prompts_per_step': 150},
                                 'mcqa': {'teacher': 'four_tasks', 'prompts_per_step': 270, 'eval_prompts': 10},
                                 'nvidia_multichallenge': {'teacher': 'multiturn', 'prompts_per_step': 12,
                                                           'max_response_tokens': 4096}}}
    path = root / 'distill.json'
    path.write_text(json.dumps(config))
    return path


class DistillConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def edit(self, path, change):
        config = json.loads(path.read_text())
        change(config)
        path.write_text(json.dumps(config))

    def test_a_config_plans_tasks_teachers_and_batch(self):
        spec = distill.load(make_config(self.root))
        tasks = spec['tasks']
        self.assertEqual(list(tasks), ['gsm8k_train', 'mcqa', 'nvidia_multichallenge'])
        # Caps default to the task's MixRL cap; eval to all of its rl_val prompts.
        self.assertEqual([t['max_response_tokens'] for t in tasks.values()], [1024, 2048, 4096])
        self.assertEqual([(t['pool'], t['eval_pool'], t['eval_prompts']) for t in tasks.values()],
                         [(900, 29, 'all'), (1800, 19, 10), (300, 29, 'all')])
        self.assertEqual(distill.check_batch(spec, 1, 6), 432)
        plan = distill.servers(spec, [6, 7], 8100)
        self.assertEqual([(p['name'], p['gpu'], p['port'], p['memory'], p['tasks']) for p in plan],
                         [('four_tasks', 6, 8100, 0.85, ['gsm8k_train', 'mcqa']),
                          ('multiturn', 7, 8101, 0.85, ['nvidia_multichallenge'])])
        text = distill.table(spec, 100, 1, 6)
        self.assertIn('3 tasks; 432 prompts per step x 1 answers = 432 answers; 2 teacher servers on GPUs 6,7', text)
        self.assertIn('note: gsm8k_train can use its 900 prompts up to 16.7 times', text)
        self.assertNotIn('error:', text)

    def test_every_problem_is_reported_with_its_field(self):
        path = make_config(self.root, tasks={'gsm8k_train': {'teacher': 'four_tasks', 'prompts_per_step': 6},
                                              'apps': {'teacher': 'nobody', 'prompts_per_step': 0},
                                              'unknown_task': {'teacher': 'multiturn', 'prompts_per_step': 6}})
        self.edit(path, lambda c: c['student'].update(hf='zoro3/hf'))
        with self.assertRaises(ValueError) as caught:
            distill.load(path)
        message = str(caught.exception)
        for part in ("student.hf: give the full path (starting with /), got 'zoro3/hf'",
                     "task apps: teacher 'nobody' is not in \"teachers\"", 'task apps: "prompts_per_step" must be',
                     'task unknown_task: "max_response_tokens" must be a positive integer'):
            self.assertIn(part, message)
        self.edit(path, lambda c: (c['student'].update(hf=str(self.root / 'zoro3' / 'hf')),
                                   c['tasks'].pop('apps'), c['tasks'].pop('unknown_task'),
                                   c['tasks'].update(calendar={'teacher': 'multiturn', 'prompts_per_step': 6})))
        with self.assertRaisesRegex(ValueError, 'task calendar: 0 rl_train and 0 rl_val prompts'):
            distill.load(path)
        self.edit(path, lambda c: c.update(splits=str(self.root / 'missing')))
        with self.assertRaisesRegex(ValueError, 'missing/rl_train.jsonl not found'):
            distill.load(path)
        self.edit(path, lambda c: c.pop('splits'))
        with self.assertRaisesRegex(ValueError, 'expected exactly'):
            distill.load(path)

    def test_teachers_must_match_the_student_and_may_share_a_checkpoint(self):
        path = make_config(self.root)
        checkpoint(self.root / 'teachers' / 'multiturn', dict(CONFIG, transformers_version='4.58.0'))  # volatile: fine
        distill.load(path)
        checkpoint(self.root / 'teachers' / 'multiturn', dict(CONFIG, hidden_size=128), template='{{ other }}')
        with self.assertRaises(ValueError) as caught:
            distill.load(path)
        self.assertIn("teacher multiturn: architecture differs from the student in config.json: ['hidden_size']",
                      str(caught.exception))
        self.assertIn('chat_template.jinja differs', str(caught.exception))
        distill.load(path, check_teachers=False)  # the trainer container: teachers are not mounted
        # The teacher = student check: both teachers are the student, still two servers.
        student = str(self.root / 'zoro3' / 'hf')
        self.edit(path, lambda c: c.update(teachers={'four_tasks': student, 'multiturn': student}))
        plan = distill.servers(distill.load(path), [6, 7], 8100)
        self.assertEqual([(p['name'], str(p['path'])) for p in plan], [('four_tasks', student), ('multiturn', student)])
        self.edit(path, lambda c: c['teachers'].update(spare=student))
        with self.assertRaisesRegex(ValueError, 'teacher\\(s\\) spare score no task'):
            distill.load(path)

    def test_servers_share_gpus_with_cumulative_memory(self):
        teachers = {f't{i}': str(self.root / 'teachers' / 'four_tasks') for i in range(3)}
        tasks = {'gsm8k_train': {'teacher': 't0', 'prompts_per_step': 6}, 'mcqa': {'teacher': 't1', 'prompts_per_step': 6},
                 'nvidia_multichallenge': {'teacher': 't2', 'prompts_per_step': 6}}
        spec = distill.load(make_config(self.root, tasks=tasks, teachers=teachers))
        plan = distill.servers(spec, [6, 7], 8100, memory=0.8)
        # SGLang sizes its cache from the memory left: the k-th of n servers on a GPU asks for k/n of the share.
        self.assertEqual([(p['gpu'], p['memory']) for p in plan], [(6, 0.4), (7, 0.8), (6, 0.8)])
        crowded = dict(spec, teachers={f't{i}': spec['teachers']['t0'] for i in range(4)})
        with self.assertRaisesRegex(ValueError, '4 teachers do not fit 1 teacher GPUs'):
            distill.servers(crowded, [6], 8100)
        self.assertEqual(distill.parse_urls('four_tasks=http://127.0.0.1:8100/,multiturn=http://127.0.0.1:8101'),
                         {'four_tasks': 'http://127.0.0.1:8100', 'multiturn': 'http://127.0.0.1:8101'})
        for bad in ('a=8100', 'a b=http://x:1', 'http://x:1'):
            with self.assertRaisesRegex(ValueError, 'MIXRL_TEACHER_URLS'):
                distill.parse_urls(bad)

    def test_eval_summary_weighs_answers_equally_and_shows_the_teacher(self):
        answers = {'gsm8k_train': [{'tokens': 2, 'stopped': True, 'gaps': [0.1, 0.3]},
                                   {'tokens': 4, 'stopped': False, 'gaps': [0., 0., 0., 6.]}],
                   'mcqa': [{'tokens': 1, 'stopped': True, 'gaps': [-0.2]}]}
        teacher = {'gsm8k_train': distill.answer_stats([{'tokens': 3, 'stopped': True}, {'tokens': 5, 'stopped': True}])}
        summary = distill.eval_summary(answers, teacher, clip=5.)
        x = summary['domains']['gsm8k_train']
        self.assertAlmostEqual(x['kl'], (0.2 + 1.5) / 2)  # mean of each answer's mean, not of all tokens
        self.assertAlmostEqual(x['clipped'], 1 / 6)
        self.assertEqual((x['answers'], x['length_mean'], x['stop_rate']), (2, 3, 0.5))
        self.assertAlmostEqual(x['length_p99'], 2 + 2 * 0.99)
        self.assertEqual((x['teacher_length_mean'], x['teacher_stop_rate']), (4, 1.))
        self.assertNotIn('teacher_answers', x)
        self.assertNotIn('teacher_length_mean', summary['domains']['mcqa'])
        self.assertAlmostEqual(summary['kl'], (0.85 - 0.2) / 2)  # tasks weigh equally

    def test_command_line_preview_plan_and_paths(self):
        path = make_config(self.root)
        run = lambda *extra: subprocess.run([sys.executable, '-m', 'slime_plugins.chimera_mixrl.distill', str(path), *extra],
                                            capture_output=True, text=True, cwd=REPO)
        ok = run('--steps', '10', '--policy-gpus', '6')
        self.assertEqual(ok.returncode, 0, ok.stderr)
        self.assertIn('nvidia_multichallenge', ok.stdout)
        self.assertEqual(run('--policy-gpus', '5').returncode, 1)  # 432 answers do not divide by 5
        plan = [line.split('\t') for line in run('--plan', '--teacher-gpus', '3', '--teacher-memory', '0.8').stdout.splitlines()]
        self.assertEqual([line[:4] for line in plan], [['8100', '3', '0.4', 'four_tasks'], ['8101', '3', '0.8', 'multiturn']])
        self.assertEqual(plan[1][4:], [str(self.root / 'teachers' / 'multiturn'), 'http://127.0.0.1:8101'])
        self.assertEqual(run('--paths').stdout.split(), [str(self.root / 'zoro3' / 'hf'), str(self.root / 'zoro3' / 'mcore'),
                                                         str(self.root / 'data' / 'splits')])
        (self.root / 'teachers' / 'multiturn' / 'config.json').unlink()
        broken = run()
        self.assertEqual(broken.returncode, 1)
        self.assertIn('teacher multiturn:', broken.stderr)

    def test_the_pushed_config_is_well_formed(self):
        config = json.loads((REPO / 'mixrl' / 'distill.json').read_text())
        self.assertEqual(set(config) - {'_readme'}, distill.FIELDS)
        paths = [config['student']['hf'], config['student']['mcore'], config['splits'], *config['teachers'].values()]
        self.assertTrue(all(p.startswith('/') for p in paths))
        self.assertTrue(all({'teacher', 'prompts_per_step'} <= set(t) <= distill.TASK_FIELDS for t in config['tasks'].values()))
        self.assertEqual(sum(t['prompts_per_step'] for t in config['tasks'].values()) % 6, 0)


if __name__ == '__main__':
    unittest.main()
