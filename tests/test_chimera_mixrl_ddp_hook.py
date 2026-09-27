"""Exercise the actual hook-install block without importing the GPU stack."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


class DDPHookTests(unittest.TestCase):
    def test_repeated_train_accepts_own_hook_but_rejects_custom(self):
        path = Path(__file__).resolve().parents[1] / 'slime/backends/megatron_utils/model.py'
        module = ast.parse(path.read_text())
        train = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'train')
        block = next(n for n in train.body if isinstance(n, ast.If)
                     and ast.unparse(n.test) == 'isinstance(model[0], DDP) and args.overlap_grad_reduce')
        code = compile(ast.Module(body=[block], type_ignores=[]), str(path), 'exec')
        class DDP:
            def no_sync(self):
                pass
        for chunks in (1, 2):
            config = SimpleNamespace(no_sync_func=None)
            models = [DDP() for _ in range(chunks)]
            scope = dict(config=config, model=models, DDP=DDP,
                         args=SimpleNamespace(overlap_grad_reduce=True, align_grad_reduce=False))
            exec(code, scope)
            original = config.no_sync_func
            exec(code, scope)
            self.assertEqual(config.no_sync_func, original)
            config.no_sync_func = lambda: None
            with self.assertRaises(AssertionError):
                exec(code, scope)


if __name__ == '__main__':
    unittest.main()
