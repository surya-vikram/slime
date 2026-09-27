import unittest
from types import SimpleNamespace

from slime_plugins.chimera_mixrl.routes import ROUTES, evaluation_summary, validate_route


class RouteTests(unittest.TestCase):
    def test_all_routes_and_history(self):
        self.assertEqual(len(ROUTES), 16)
        self.assertEqual(len({d for d, _ in ROUTES.values()}), 9)
        for task, (domain, verifier) in ROUTES.items():
            row = dict(task=task, domain=domain, verifier=verifier, binary=verifier != 'quality',
                       messages=[{'role': 'user', 'content': 'Earlier'},
                                 {'role': 'assistant', 'content': 'Earlier answer'},
                                 {'role': 'user', 'content': 'Follow-up'}])
            validate_route(row)
            with self.assertRaises(ValueError):
                validate_route(dict(row, turns=[{}]))

    def test_equal_domain_not_equal_sample_aggregate_and_binary_only_pass(self):
        def samples(scores, binary):
            return [SimpleNamespace(metadata={'grade': {'status': 'valid', 'score': s,
                        'passed': bool(s) if binary else None}}, status=SimpleNamespace(name='COMPLETED'))
                    for s in scores]
        groups = [({'task': 'mcqa', 'domain': 'knowledge', 'binary': True}, samples([1, 0, 0, 0], True))]
        groups += [({'task': 'cascade_chat', 'domain': 'quality', 'binary': False},
                    samples([.75] * 4, False)) for _ in range(3)]
        result = evaluation_summary(groups)
        self.assertEqual(result['equal_domain_mean'], .5)
        self.assertEqual(result['tasks']['mcqa']['pass@4'], 1)
        self.assertNotIn('pass@4', result['tasks']['cascade_chat'])


if __name__ == '__main__':
    unittest.main()
