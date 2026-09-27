"""Execute the actual synchronous loop with lightweight boundary doubles."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch


class SkipTests(unittest.TestCase):
    def harness(self, start=0, end=20, constant=False, budget=None, baseline=False,
                periodic=False, explicit=False):
        path = Path(__file__).resolve().parents[1] / 'train.py'
        node = next(n for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'train')
        actor, manager = MagicMock(), MagicMock()
        manager.generate.remote.return_value = {'skip_optimizer': True} if constant else 'batch'
        args = SimpleNamespace(release_train=False, offload_rollout=True, offload_train=True,
            use_critic=False, check_weight_update_equal=False, num_rollout=end, start_rollout_id=start,
            eval_interval=8, skip_eval_before_train=not baseline, save_interval=8, rollout_global_dataset=True)
        env = dict(ray=SimpleNamespace(get=lambda x: x), configure_logger=lambda: None,
            init_tracking=lambda _: None, finish_tracking=lambda _: None,
            create_placement_groups=lambda _: {'rollout': None},
            create_rollout_manager=lambda *a: (manager, 100),
            create_training_models=lambda *a: (actor, None), should_run_periodic_action=lambda *a: periodic)
        env['_explicit_eval_due'] = lambda _: explicit
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), env)
        with patch('slime_plugins.chimera_mixrl.budget.RunBudget.from_env', return_value=budget):
            env['train'](args)
        return actor, manager

    def test_explicit_update_evaluation_runs_off_periodic_cadence(self):
        budget = MagicMock()
        budget.should_stop.return_value = budget.finish_update.return_value = False
        _, manager = self.harness(start=0, end=2, budget=budget, explicit=True)
        self.assertEqual(manager.eval.remote.call_count, 2)
        manager.eval.remote.assert_any_call(0, final=False)

    def test_deadline_after_optimizer_saves_exact_completed_boundary_and_evaluates(self):
        budget = MagicMock()
        budget.should_stop.return_value = False
        budget.finish_update.return_value = True
        actor, manager = self.harness(start=8, budget=budget)
        actor.async_train.assert_called_once_with(8, 'batch')
        actor.save_model.assert_called_once_with(8, force_sync=True)
        manager.save.remote.assert_called_once_with(8)
        manager.eval.remote.assert_called_once_with(8, final=True)
        self.assertEqual(actor.update_weights.call_count, 2)
        manager.generate.remote.assert_called_once_with(8)

    def test_constant_checkpoint_rebroadcasts_weights_before_eval(self):
        budget = MagicMock()
        budget.should_stop.return_value = False
        budget.finish_update.return_value = True
        actor, manager = self.harness(start=8, budget=budget, constant=True)
        self.assertEqual(actor.update_weights.call_count, 2)
        actor.async_train.assert_not_called()
        manager.generate.remote.assert_called_once_with(8)

    def test_deadline_during_baseline_never_consumes_training_batch(self):
        budget = MagicMock()
        budget.should_stop.side_effect = [False, True]
        actor, manager = self.harness(budget=budget, baseline=True)
        actor.async_train.assert_not_called()
        actor.save_model.assert_not_called()
        manager.generate.remote.assert_not_called()
        manager.eval.remote.assert_called_once_with(0)

    def test_budget_exhausted_after_periodic_eval_saves_prior_not_next_rollout(self):
        budget = MagicMock()
        budget.should_stop.side_effect = [False, True]
        budget.finish_update.return_value = False
        actor, manager = self.harness(start=8, budget=budget, periodic=True)
        self.assertEqual(actor.save_model.call_args.args, (8,))
        self.assertEqual(actor.save_model.call_args.kwargs, {'force_sync': True})
        manager.generate.remote.assert_called_once_with(8)
        manager.eval.remote.assert_called_once_with(8, final=False)

    def test_constant_batch_deadline_saves_sampler_without_optimizer(self):
        budget = MagicMock()
        budget.should_stop.return_value = False
        budget.finish_update.return_value = True
        actor, manager = self.harness(start=8, budget=budget, constant=True)
        actor.async_train.assert_not_called()
        actor.save_model.assert_called_once_with(8, force_sync=True)
        manager.save.remote.assert_called_once_with(8)
        manager.eval.remote.assert_called_once_with(8, final=True)

    def test_normal_final_update_evaluates_even_off_cadence(self):
        budget = MagicMock()
        budget.should_stop.return_value = budget.finish_update.return_value = False
        actor, manager = self.harness(start=8, end=9, budget=budget)
        manager.eval.remote.assert_called_once_with(8, final=True)

    def test_empty_signal_skips_training_but_saves_sampler_and_evaluates(self):
        path = Path(__file__).resolve().parents[1] / 'train.py'
        node = next(n for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'train')
        actor, manager = MagicMock(), MagicMock()
        manager.generate.remote.return_value = {'skip_optimizer': True}
        args = SimpleNamespace(release_train=False, offload_rollout=True,
            check_weight_update_equal=False, num_rollout=2, start_rollout_id=0,
            eval_interval=1, skip_eval_before_train=True, save_interval=1,
            rollout_global_dataset=True)
        env = dict(ray=SimpleNamespace(get=lambda x: x),
            configure_logger=lambda: None, init_tracking=lambda _: None,
            finish_tracking=lambda _: None,
            create_placement_groups=lambda _: {'rollout': None},
            create_rollout_manager=lambda *a: (manager, 10),
            create_training_models=lambda *a: (actor, None),
            _explicit_eval_due=lambda _: False,
            should_run_periodic_action=lambda *a: True)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), env)
        env['train'](args)
        actor.async_train.assert_not_called()
        self.assertEqual(actor.update_weights.call_count, 3)  # initial + both checkpoint offloads
        self.assertEqual(actor.save_model.call_count, 2)
        self.assertEqual(manager.save.remote.call_count, 2)
        self.assertEqual(manager.eval.remote.call_count, 2)
        manager.dispose.remote.assert_called_once()
