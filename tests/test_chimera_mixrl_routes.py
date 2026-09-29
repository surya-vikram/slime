import unittest
from types import SimpleNamespace

from slime_plugins.chimera_mixrl import tasks
from slime_plugins.chimera_mixrl.routes import evaluation_summary, validate_route


class RouteTests(unittest.TestCase):
    def test_all_routes_and_history(self):
        routes = tasks.resolved(tasks.load())['routes']
        self.assertEqual(len(routes), 16)
        self.assertEqual(len({r['domain'] for r in routes.values()}), 9)
        for task, route in routes.items():
            row = dict(task=task, domain=route['domain'], verifier=route['verifier'],
                       binary=route['reward'] == 'binary',
                       messages=[{'role': 'user', 'content': 'Earlier'},
                                 {'role': 'assistant', 'content': 'Earlier answer'},
                                 {'role': 'user', 'content': 'Follow-up'}])
            validate_route(row, routes)
            for broken in (dict(row, turns=[{}]), dict(row, binary=not row['binary']),
                           dict(row, verifier='other'), dict(row, task='unlisted')):
                with self.assertRaises(ValueError):
                    validate_route(broken, routes)

    def test_equal_domain_not_equal_sample_aggregate_and_binary_only_pass(self):
        def samples(scores, binary):
            return [SimpleNamespace(metadata={'grade': {'status': 'valid', 'score': s,
                        'passed': bool(s) if binary else None},
                        'grading_text': '<think>\nhm</think>\nB' if s == 1 else 'A'},
                        status=SimpleNamespace(name='COMPLETED'))
                    for s in scores]
        groups = [({'task': 'mcqa', 'domain': 'knowledge', 'binary': True}, samples([1, 0, 0, 0], True))]
        groups += [({'task': 'cascade_chat', 'domain': 'quality', 'binary': False},
                    samples([.75] * 4, False)) for _ in range(3)]
        result = evaluation_summary(groups)
        self.assertEqual(result['equal_domain_mean'], .5)
        self.assertEqual(result['tasks']['mcqa']['pass@4'], 1)
        self.assertEqual(result['tasks']['mcqa']['think_rate'], .25)
        self.assertEqual(result['domains']['quality']['think_rate'], 0)
        self.assertNotIn('pass@4', result['tasks']['cascade_chat'])


if __name__ == '__main__':
    unittest.main()
