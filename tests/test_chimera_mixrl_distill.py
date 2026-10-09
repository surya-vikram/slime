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


if __name__ == '__main__':
    unittest.main()
