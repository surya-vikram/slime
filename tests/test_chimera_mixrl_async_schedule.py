"""Run upstream async loop with deterministic fake actors; GPU overlap is separate."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest


class AsyncScheduleTests(unittest.TestCase):
    def test_one_batch_lookahead_and_final_drained_save(self):
        path = Path(__file__).resolve().parents[1] / 'train_async.py'
        function = next(n for n in ast.parse(path.read_text()).body
                        if isinstance(n, ast.FunctionDef) and n.name == 'train')
        events, behavior = [], []
        state = {'trained': 0, 'served': 0}
        def generate(i):
            events.append(('generate', i, state['served']))
            return (i, state['served'])
        def train(i, batch):
            self.assertEqual(batch[0], i)
            behavior.append(batch[1])
            events.append(('train', i))
            state['trained'] += 1
        def update():
            state['served'] = state['trained']
            events.append(('update', state['served']))
        def remote(fn):
            return NS(remote=fn)
        manager = NS(generate=remote(generate), save=remote(lambda i: events.append(('sampler_save', i))),
                     eval=remote(lambda i: events.append(('eval', i))), dispose=remote(lambda: None))
        actor = NS(update_weights=update, async_train=train,
                   save_model=lambda i, force_sync: events.append(('save', i, force_sync)))
        noop = lambda *a, **kw: None
        scope = dict(ray=NS(get=lambda value: value), configure_logger=noop, init_tracking=noop,
                     finish_tracking=noop, create_placement_groups=lambda args: {'rollout': None},
                     create_rollout_manager=lambda *a: (manager, None),
                     create_training_models=lambda *a: (actor, None),
                     should_run_periodic_action=lambda i, interval, epoch, total=None:
                     (i + 1) % interval == 0 or (total is not None and i + 1 == total))
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), scope)
        args = NS(colocate=False, release_train=False, check_weight_update_equal=False,
                  start_rollout_id=0, num_rollout=3, use_critic=False, save_interval=3,
                  rollout_global_dataset=True, update_weights_interval=1, eval_interval=1)
        scope['train'](args)
        self.assertEqual(behavior, [0, 0, 1])
        self.assertLess(events.index(('generate', 1, 0)), events.index(('train', 0)))
        self.assertEqual([e for e in events if e[0] == 'save'], [('save', 2, True)])
        self.assertEqual([e for e in events if e[0] == 'sampler_save'], [('sampler_save', 2)])


if __name__ == '__main__':
    unittest.main()
