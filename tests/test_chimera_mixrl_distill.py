"""Distillation inputs (slime_plugins/chimera_mixrl/distill.py): folder checks, batch, teacher placement, preview."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from slime_plugins.chimera_mixrl import distill

CONFIG = {'architectures': ['ChimeraForCausalLM'], 'hidden_size': 64, 'num_hidden_layers': 2, 'vocab_size': 100,
          'transformers_version': '4.57.0'}


def checkpoint(path, config=CONFIG, template='{{ messages }}'):
    path.mkdir(parents=True, exist_ok=True)
    (path / 'config.json').write_text(json.dumps(config))
    (path / 'tokenizer.json').write_text('{"model": "fixture"}')
    (path / 'chat_template.jinja').write_text(template)
    (path / 'model.safetensors').write_bytes(b'weights')


def prompts(path, n, prefix, last_role='user'):
    path.write_text(''.join(json.dumps({'id': f'{prefix}-{i}', 'messages': [{'role': 'system', 'content': 'Be brief.'},
                                                                           {'role': last_role, 'content': 'q' * 40}]}) + '\n'
                            for i in range(n)))


class DistillFolderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        checkpoint(self.root / 'student' / 'hf')
        (self.root / 'student' / 'mcore').mkdir(parents=True)
        (self.root / 'student' / 'mcore' / 'latest_checkpointed_iteration.txt').write_text('100')
        self.domain('teacher_x', 9000)
        # teacher_y's answers run longer, so its teacher has more tokens to read per step.
        self.domain('teacher_y', 12000, {'prompts_per_step': 264, 'max_response_tokens': 4096})

    def tearDown(self):
        self.temp.cleanup()

    def domain(self, name, pool, settings=None, teacher=True):
        base = self.root / 'domains' / name
        base.mkdir(parents=True)
        prompts(base / 'prompts.jsonl', pool, f'{name}-train')
        prompts(base / 'eval.jsonl', 32, f'{name}-eval')
        (base / 'domain.json').write_text(json.dumps(settings or {'prompts_per_step': 264}))
        if teacher:
            checkpoint(base / 'teacher')
        return base

    def test_two_teachers_plan_a_528_prompt_batch_one_teacher_per_gpu(self):
        spec = distill.load(self.root)
        self.assertEqual(list(spec['domains']), ['teacher_x', 'teacher_y'])
        self.assertEqual(distill.batch_prompts(spec), 528)
        self.assertEqual(distill.check_batch(spec, 1, 6), 528)
        plan = distill.placement(spec, [6, 7], 8100)
        # The heavier teacher is placed first: its own GPU, the first port.
        self.assertEqual([(p['name'], p['gpu'], p['port']) for p in plan], [('teacher_y', 6, 8100), ('teacher_x', 7, 8101)])
        self.assertAlmostEqual(distill.max_passes(spec['domains']['teacher_x'], 100), 264 * 100 / 9000)
        text = distill.table(spec, 100, 1, 6, (6, 7), 8100)
        self.assertIn('2 of 2 domains enabled; 528 prompts per step x 1 answers = 528 answers; 2 teacher servers', text)
        self.assertIn('max passes in 100 steps', text)
        self.assertNotIn('note:', text)
        self.assertNotIn('error:', text)

    def test_every_problem_is_reported_by_domain(self):
        base = self.root / 'domains'
        (base / 'teacher_x' / 'domain.json').write_text(json.dumps({'max_response_tokens': 1024}))
        prompts(base / 'teacher_y' / 'prompts.jsonl', 5, 'teacher_y-train', last_role='assistant')
        (self.root / 'student' / 'mcore' / 'latest_checkpointed_iteration.txt').unlink()
        with self.assertRaises(ValueError) as caught:
            distill.load(self.root)
        message = str(caught.exception)
        self.assertIn('student Megatron checkpoint', message)
        self.assertIn('"prompts_per_step" is required', message)
        self.assertIn('the last message must be from the user', message)

    def test_rows_ids_and_held_out_eval(self):
        base = self.root / 'domains' / 'teacher_x'
        with open(base / 'prompts.jsonl', 'a') as stream:
            stream.write(json.dumps({'id': 'teacher_x-train-0', 'messages': [{'role': 'user', 'content': 'x'}]}) + '\n')
            stream.write('{not json\n')
            stream.write(json.dumps({'id': 'extra', 'messages': [{'role': 'user', 'content': 'x'}], 'answer': 4}) + '\n')
        with self.assertRaises(ValueError) as caught:
            distill.load(self.root)
        for part in ('duplicate id', 'not JSON', 'unknown fields'):
            self.assertIn(part, str(caught.exception))
        prompts(base / 'prompts.jsonl', 100, 'same')
        prompts(base / 'eval.jsonl', 10, 'same')
        with self.assertRaisesRegex(ValueError, 'eval must be held out'):
            distill.load(self.root)

    def test_teacher_must_share_architecture_tokenizer_and_template(self):
        teacher = self.root / 'domains' / 'teacher_y' / 'teacher'
        checkpoint(teacher, dict(CONFIG, transformers_version='4.58.0'))  # a volatile key: fine
        distill.load(self.root)
        checkpoint(teacher, dict(CONFIG, hidden_size=128), template='{{ other }}')
        with self.assertRaises(ValueError) as caught:
            distill.load(self.root)
        self.assertIn("architecture differs from the student in config.json: ['hidden_size']", str(caught.exception))
        self.assertIn('chat_template.jinja differs', str(caught.exception))

    def test_a_shared_teacher_folder_is_served_once(self):
        base = self.domain('teacher_z', 9000, teacher=False)
        os.symlink(self.root / 'domains' / 'teacher_x' / 'teacher', base / 'teacher')
        spec = distill.load(self.root)
        plan = distill.placement(spec, [6, 7], 8100)
        self.assertEqual(sorted(p['name'] for p in plan), ['teacher_x+teacher_z', 'teacher_y'])

    def test_packing_by_load_and_capacity(self):
        for i in range(2):
            self.domain(f'teacher_{i}', 9000, {'prompts_per_step': 12})
        # teacher_y alone outweighs the other three, so it keeps GPU 6 and they share GPU 7.
        plan = {p['name']: p['gpu'] for p in distill.placement(distill.load(self.root), [6, 7], 8100)}
        self.assertEqual(plan, {'teacher_y': 6, 'teacher_x': 7, 'teacher_0': 7, 'teacher_1': 7})
        # GPU 7 is full (3 per GPU): the next teacher goes to GPU 6 despite its load.
        self.domain('teacher_2', 9000, {'prompts_per_step': 12})
        plan = {p['name']: p['gpu'] for p in distill.placement(distill.load(self.root), [6, 7], 8100)}
        self.assertEqual(sorted(plan.values()), [6, 6, 7, 7, 7])
        for i in range(3, 5):
            self.domain(f'teacher_{i}', 9000, {'prompts_per_step': 12})
        with self.assertRaisesRegex(ValueError, '7 teachers do not fit 2 teacher GPUs'):
            distill.placement(distill.load(self.root), [6, 7], 8100)

    def test_batch_divisibility_and_reuse_warning(self):
        (self.root / 'domains' / 'teacher_x' / 'domain.json').write_text(json.dumps({'prompts_per_step': 301}))
        spec = distill.load(self.root)
        with self.assertRaisesRegex(ValueError, 'must divide by the 6 student GPUs'):
            distill.check_batch(spec, 1, 6)
        text = distill.table(spec, 100, 1, 6, (6, 7), 8100)
        self.assertIn('error: 565 answers do not divide by the 6 student GPUs', text)
        self.assertIn('note: teacher_x can use its 9000 prompts up to 3.3 times in 100 steps', text)

    def test_disabled_domain_needs_no_matching_teacher(self):
        base = self.domain('old', 100, {'prompts_per_step': 6, 'enabled': False}, teacher=False)
        checkpoint(base / 'teacher', dict(CONFIG, hidden_size=999))
        spec = distill.load(self.root)
        self.assertEqual(sorted(distill.enabled(spec)), ['teacher_x', 'teacher_y'])

    def test_command_line_exit_codes(self):
        run = lambda *extra: subprocess.run([sys.executable, '-m', 'slime_plugins.chimera_mixrl.distill', str(self.root), *extra],
                                            capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1])
        ok = run('--steps', '100', '--policy-gpus', '6')
        self.assertEqual(ok.returncode, 0, ok.stderr)
        self.assertIn('teacher_x', ok.stdout)
        self.assertEqual(run('--policy-gpus', '5').returncode, 1)
        shutil.rmtree(self.root / 'domains' / 'teacher_y' / 'teacher')
        broken = run()
        self.assertEqual(broken.returncode, 1)
        self.assertIn('domain teacher_y: missing teacher/', broken.stderr)

    def test_servers_split_memory_and_name_their_domains(self):
        for i in range(2):
            self.domain(f'teacher_{i}', 9000, {'prompts_per_step': 12})
        plan = distill.servers(distill.load(self.root), [6, 7], 8100, memory=0.9, host='10.0.0.1')
        by_gpu = {}
        for p in plan:
            by_gpu.setdefault(p['gpu'], []).append(p['memory'])
        # SGLang sizes its cache from the memory left: the k-th of n servers on a GPU asks for k/n of the share.
        self.assertEqual(by_gpu, {6: [0.9], 7: [0.3, 0.6, 0.9]})
        self.assertEqual(plan[0]['url'], 'http://10.0.0.1:8100')
        urls = distill.parse_urls(','.join(f'{p["name"]}={p["url"]}' for p in plan) + ',')
        self.assertEqual(sorted(urls), sorted(p['name'] for p in plan))
        shared = distill.parse_urls('a+b=http://127.0.0.1:8100/')
        self.assertEqual(shared, {'a+b': {'url': 'http://127.0.0.1:8100', 'domains': ['a', 'b']}})
        for bad in ('a=8100', 'a b=http://x:1', 'http://x:1'):
            with self.assertRaisesRegex(ValueError, 'MIXRL_TEACHER_URLS'):
                distill.parse_urls(bad)

    def test_the_trainer_reads_the_folder_without_the_teachers(self):
        # In the trainer container a teacher/ symlink may point outside the mounted folder.
        teacher = self.root / 'domains' / 'teacher_x' / 'teacher'
        shutil.rmtree(teacher)
        os.symlink(self.root / 'elsewhere', teacher)
        with self.assertRaisesRegex(ValueError, 'missing teacher/'):
            distill.load(self.root)
        spec = distill.load(self.root, check_teachers=False)
        self.assertEqual(spec['domains']['teacher_x']['teacher'], self.root / 'elsewhere')
        rows = distill.read_rows(spec['domains']['teacher_x']['eval'])
        self.assertEqual((len(rows), rows[0]['id']), (32, 'teacher_x-eval-0'))
        self.domain('bad+name', 10)
        with self.assertRaisesRegex(ValueError, 'domain bad\\+name: use letters'):
            distill.load(self.root)

    def test_eval_summary_weighs_answers_equally_and_shows_the_teacher(self):
        answers = {'teacher_x': [{'tokens': 2, 'stopped': True, 'gaps': [0.1, 0.3]},
                                 {'tokens': 4, 'stopped': False, 'gaps': [0., 0., 0., 6.]}],
                   'teacher_y': [{'tokens': 1, 'stopped': True, 'gaps': [-0.2]}]}
        teacher = {'teacher_x': distill.answer_stats([{'tokens': 3, 'stopped': True}, {'tokens': 5, 'stopped': True}])}
        summary = distill.eval_summary(answers, teacher, clip=5.)
        x = summary['domains']['teacher_x']
        self.assertAlmostEqual(x['kl'], (0.2 + 1.5) / 2)  # mean of each answer's mean, not of all tokens
        self.assertAlmostEqual(x['clipped'], 1 / 6)
        self.assertEqual((x['answers'], x['length_mean'], x['stop_rate']), (2, 3, 0.5))
        self.assertAlmostEqual(x['length_p99'], 2 + 2 * 0.99)
        self.assertEqual((x['teacher_length_mean'], x['teacher_stop_rate']), (4, 1.))
        self.assertNotIn('teacher_answers', x)
        self.assertNotIn('teacher_length_mean', summary['domains']['teacher_y'])
        self.assertAlmostEqual(summary['kl'], (0.85 - 0.2) / 2)  # domains weigh equally

    def test_plan_lines_for_the_teacher_script(self):
        result = subprocess.run([sys.executable, '-m', 'slime_plugins.chimera_mixrl.distill', str(self.root), '--plan',
                                 '--teacher-gpus', '3', '--teacher-port', '9000', '--teacher-memory', '0.8'],
                                capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1])
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = [line.split('\t') for line in result.stdout.splitlines()]
        self.assertEqual([line[:4] for line in lines], [['9000', '3', '0.4', 'teacher_y'], ['9001', '3', '0.8', 'teacher_x']])
        self.assertEqual(lines[1][4:], [str(self.root / 'domains' / 'teacher_x' / 'teacher'), 'http://127.0.0.1:9001'])


if __name__ == '__main__':
    unittest.main()
