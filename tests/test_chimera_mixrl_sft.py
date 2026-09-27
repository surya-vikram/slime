import unittest

from slime_plugins.chimera_mixrl.sft import demonstration, tokenize_final, validate_provenance


class SFTTests(unittest.TestCase):
    def test_published_provenance_does_not_require_invented_judge_grade(self):
        metadata = dict(split='rl_train', family_id='family', verification_status='publisher_reference',
            reference_source={'repo':'publisher/dataset','revision':'a'*40,'row_hash':'b'*64})
        validate_provenance(metadata)
        for broken in (dict(metadata, split='main_test'), dict(metadata, reference_source={}),
                       dict(metadata, verification_status=None)):
            with self.assertRaises(ValueError):
                validate_provenance(broken)

    def setUp(self):
        self.row = dict(id='train-1', family_id='family-1', task='mcqa', binary=True,
            messages=[{'role': 'user', 'content': 'Earlier'},
                      {'role': 'assistant', 'content': 'History'},
                      {'role': 'user', 'content': 'Answer?'}])
        self.response = {'text': 'B', 'finish_reason': 'stop'}
        self.grade = {'status': 'valid', 'score': 1., 'passed': True}

    def test_provenance_and_final_turn_mask(self):
        demo = demonstration(self.row, self.response, self.grade, 'pinned', set())
        self.assertEqual([m['step_loss_mask'] for m in demo['messages']], [0, 0, 0, 1])
        self.assertNotIn('step_loss_mask', self.row['messages'][0])
        self.assertEqual(demo['metadata']['parent_id'], 'train-1')

    def test_reject_leakage_caps_and_bad_grades(self):
        for response, grade, heldout in (
            (self.response, self.grade, {'family-1'}),
            (dict(self.response, finish_reason='length'), self.grade, set()),
            (self.response, dict(self.grade, status='error'), set()),
            (self.response, dict(self.grade, score=float('nan')), set()),
            (self.response, dict(self.grade, passed=False), set()),
        ):
            with self.assertRaises(ValueError):
                demonstration(self.row, response, grade, 'pinned', heldout)

    def test_quality_requires_acceptability_not_only_high_score(self):
        row = dict(self.row, binary=False)
        with self.assertRaises(ValueError):
            demonstration(row, self.response, self.grade, 'pinned', set())
        demonstration(row, self.response, dict(self.grade, components={'acceptability': True}), 'pinned', set())

    def test_exact_prefix_eos_and_context(self):
        class Tokenizer:
            eos_token_id = 9
            def apply_chat_template(self, messages, add_generation_prompt, **kwargs):
                return [1, 2] if add_generation_prompt else [1, 2, 3, 9]
        messages = demonstration(self.row, self.response, self.grade, 'pinned', set())['messages']
        self.assertEqual(tokenize_final(Tokenizer(), messages, {}, 4), ([1, 2, 3, 9], [0, 0, 1, 1]))
        with self.assertRaises(ValueError):
            tokenize_final(Tokenizer(), messages, {}, 3)
