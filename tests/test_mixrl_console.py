"""mixrl/internal/console.py (the concise terminal view) and the MIXRL_STEP clock."""
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from slime_plugins.chimera_mixrl.steplog import StepClock

spec = importlib.util.spec_from_file_location('console', Path(__file__).resolve().parents[1] / 'mixrl/internal/console.py')
console = importlib.util.module_from_spec(spec)
spec.loader.exec_module(console)

COLLECTION = {'rollout_id': 0, 'routes': {
    'gsm8k_train': {'attempted': 10, 'accepted': 6, 'padding': 2, 'refilled': 4, 'responses': 160, 'capped': 8,
                    'raw_reward_mean': 0.5, 'grade_failed': 0},
    'mcqa': {'attempted': 4, 'accepted': 4, 'padding': 0, 'refilled': 0, 'responses': 64, 'capped': 0,
             'raw_reward_mean': 0.25, 'grade_failed': 1, 'masked': 2}}}


class Clock:
    def __init__(self):
        self.now = 1_000_000.0

    def __call__(self):
        return self.now


def make(full=False, progress=60):
    out, metrics, clock = io.StringIO(), io.StringIO(), Clock()
    return console.Console(out, None, metrics, offset=100, full=full, progress_seconds=progress, clock=clock), out, metrics, clock


def lines(out):
    return [line.split('] ', 1)[1] for line in out.getvalue().splitlines()]


def pipeline(seconds, **extra):
    return 'MIXRL_PIPELINE ' + json.dumps({'rollout_id': 0, 'phase': 'train', 'seconds': seconds, 'gen_queued': 0,
                                           'generating': 50, 'grade_queued': 0, 'grading': 5, 'done': 10,
                                           'gen_tokens_per_s': 1000, 'loop_lag_ms': 1,
                                           'sglang': {'running': 50, 'waiting': 0, 'kv_usage': 0.4},
                                           'judge': {'running': 5, 'waiting': 0, 'kv_usage': 0.1}, **extra})


class ConsoleTests(unittest.TestCase):
    def test_step_lines_carry_step_time_eta_training_and_rollout_metrics(self):
        c, out, metrics, clock = make()
        c.feed('(RolloutManager pid=7) ' + pipeline(15) + '\n')
        clock.now += 15
        c.feed('(RolloutManager pid=7) ' + pipeline(30, sglang={'running': 80, 'waiting': 0, 'kv_usage': 0.7}) + '\n')
        c.feed('(RolloutManager pid=7) MIXRL_TIMING ' + json.dumps({'rollout_id': 0, 'generated_tokens': 44800,
                                                                    'generated_tokens_per_second': 3000}) + '\n')
        c.feed('(RolloutManager pid=7) MIXRL_COLLECTION ' + json.dumps(COLLECTION) + '\n')
        c.feed('(MegatronTrainRayActor pid=9) MIXRL_TRAIN ' + json.dumps(
            {'rollout_id': 0, 'step_id': 0, 'loss': -0.0561, 'grad_norm': 0.154, 'entropy': 0.448, 'lr-pg_0': 5e-8,
             'train_rollout_logprob_abs_diff': 0.005, 'importance_masked_fraction': 0.01}) + '\n')
        c.feed("(MegatronTrainRayActor pid=9) [2026-10-02 11:12:05] train_metric_utils.py:390 - perf 0: "
               "{'perf/actor_train_time': 157.5, 'perf/actor_train_tok_per_s': 12345.0}\n")
        c.feed('MIXRL_STEP ' + json.dumps({'rollout_id': 0, 'num_rollout': 400, 'seconds': 1202.0, 'rollout': 644.0,
                                           'train': 251.0, 'sync': 24.0, 'save': 52.0}) + '\n')
        printed = lines(out)
        # one progress line (the second came within a minute), three step lines, and a warning: mcqa could
        # not grade 2 of its 66 responses (over 1%)
        self.assertEqual(len(printed), 5)
        head, detail, tasks = printed[1:4]
        self.assertEqual(printed[4], 'WARNING task mcqa: 2 of 66 responses could not be graded and were masked this step (over 1%)')
        for part in ('step   1/400', 'step time: 20m02s', 'ETA: 5d13h', 'reward: 0.429', 'loss: -5.6100E-02',
                     'grad norm: 0.154', 'lr: 5.00E-08', 'logprob diff: 0.0050', 'IS masked: 1%'):
            self.assertIn(part, head)
        for part in ('rollout: 10m44s', 'train: 4m11s (12.3k tok/s)', 'sync: 24s', 'save: 52s',
                     'groups: 10/12 informative, 2 padded', 'refills: 4', 'responses: 224', 'resp len: 200',
                     'capped: 4%', 'peak KV: sglang 70%, judge 10%', 'masked (ungradable): 2', 'ungraded groups: 1'):
            self.assertIn(part, detail)
        self.assertIn('gsm8k_train 0.50 6/8 | mcqa 0.25 4/4', tasks)
        kinds = [json.loads(line)['kind'] for line in metrics.getvalue().splitlines()]
        self.assertEqual(kinds, ['pipeline', 'pipeline', 'timing', 'collection', 'train', 'trainer_perf', 'step'])

    def test_progress_is_throttled_and_eval_is_labelled_by_updates(self):
        c, out, _, clock = make(progress=60)
        for seconds in (15, 30, 45, 60, 75):
            c.feed(pipeline(seconds) + '\n')
            clock.now += 15
        self.assertEqual([l.split(' | ')[1] for l in lines(out)], ['15s', '1m15s'])
        evaluation = 'MIXRL_EVAL ' + json.dumps({'rollout_id': 0, 'equal_domain_mean': 0.5,
                                                 'domains': {'math': {'mean_score': 0.6}, 'logic': {'mean_score': 0.4}}}) + '\n'
        c.feed(evaluation)  # the baseline: before step 1's rollout
        self.assertEqual(lines(out)[-1], 'eval before step 1 | score: 0.500 | math 0.600 | logic 0.400')
        c.feed('MIXRL_COLLECTION ' + json.dumps({'rollout_id': 0, 'routes': {}}) + '\n')
        c.feed(evaluation)  # after step 1, trained or skipped
        self.assertEqual(lines(out)[-1], 'eval after step 1 | score: 0.500 | math 0.600 | logic 0.400')
        # Ray forwards the rollout worker's eval line later than the driver's step line.
        c.feed('MIXRL_STEP ' + json.dumps({'rollout_id': 0, 'num_rollout': 2, 'seconds': 60.0}) + '\n')
        c.feed(evaluation)
        self.assertEqual(lines(out)[-1], 'eval after step 1 | score: 0.500 | math 0.600 | logic 0.400')

    def test_errors_warnings_and_launcher_lines(self):
        c, out, _, clock = make()
        c.feed('MIXRL_SAY MixRL run r1 (new) | model: m\n')
        c.feed('(MegatronTrainRayActor pid=9) Traceback (most recent call last):\n')
        c.feed('(MegatronTrainRayActor pid=9)   File "x.py", line 1, in f\n')
        c.feed('(MegatronTrainRayActor pid=9) torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2 GiB\n')
        for _ in range(3):
            c.feed('MIXRL_REWARD_RETRY attempt 2/12: ServiceHTTPError: HTTP 503\n')
        clock.now += 61
        c.feed('MIXRL_REWARD_RETRY attempt 3/12: ServiceHTTPError: HTTP 503\n')
        c.feed('error: TRAIN_GPUS lists 1 GPUs\n')
        c.feed("Job 'raysubmit_x' failed\n")
        c.feed('(SGLangEngine pid=3) [2026] Decode batch. #running-req: 5\n')  # verbose: train.log only
        self.assertEqual(lines(out), [
            'MixRL run r1 (new) | model: m',
            'ERROR torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2 GiB | traceback: train.log line 102',
            'WARNING reward service request failed, retrying: attempt 2/12: ServiceHTTPError: HTTP 503',
            'WARNING reward service request failed, retrying: attempt 3/12: ServiceHTTPError: HTTP 503 (and 2 more like it)',
            'ERROR error: TRAIN_GPUS lists 1 GPUs | train.log line 109',
            "Job 'raysubmit_x' failed"])

    def test_a_response_the_judge_cannot_judge_is_named_as_such(self):
        c, out, _, _ = make()
        record = {'task': 'cascade_chat', 'row': 'r1', 'ungradable': True,
                  'error': 'ServiceHTTPError: HTTP 422: Judge did not finish'}
        c.feed('MIXRL_GRADE_FAILED ' + json.dumps(record) + '\n')
        c.feed('MIXRL_GRADE_FAILED ' + json.dumps(dict(record, task='apps', ungradable=False)) + '\n')
        self.assertEqual([l.split(' | ')[0] for l in lines(out)],
                         ['WARNING judge could not judge a response (not retried), response masked',
                          'WARNING grading failed, response masked'])

    def test_full_mode_passes_every_line_and_a_lost_terminal_does_not_stop_it(self):
        c, out, metrics, _ = make(full=True)
        c.feed('(SGLangEngine pid=3) Decode batch. #running-req: 5\n')
        c.feed(pipeline(15) + '\n')
        self.assertEqual(out.getvalue().splitlines()[0], '(SGLangEngine pid=3) Decode batch. #running-req: 5')
        self.assertEqual(len(metrics.getvalue().splitlines()), 1)

        class Gone(io.StringIO):
            def write(self, text):
                raise BrokenPipeError

        with tempfile.TemporaryDirectory() as folder:
            with open(Path(folder) / 'console.log', 'w') as log:
                c = console.Console(Gone(), log, None)
                c.feed('MIXRL_SAY still logged\n')
                c.feed('MIXRL_SAY after the terminal went away\n')
            self.assertEqual(len((Path(folder) / 'console.log').read_text().splitlines()), 2)

    def test_skipped_step_and_a_malformed_record(self):
        c, out, _, _ = make()
        c.feed('MIXRL_COLLECTION {not json\n')
        c.feed('MIXRL_STEP ' + json.dumps({'rollout_id': 4, 'num_rollout': 10, 'seconds': 300.0, 'rollout': 290.0,
                                           'skipped': True}) + '\n')
        self.assertIn('step  5/10 | step time: 5m00s | ETA: 25m00s | skipped: no informative groups', lines(out)[0])


class SettingsLineTests(unittest.TestCase):
    def test_settings_come_from_the_resolved_config_and_the_train_command(self):
        config = {'tasks_file': 'mixrl/tasks.json', 'rollout_temperature': 1.0, 'rollout_top_p': 0.95, 'rollout_top_k': 20,
                  'routing_replay': 1, 'refill_rounds': 2, 'oversample': 0.3, 'response_concurrency': 3200,
                  'reward_concurrency': 1400, 'inflight_groups': 100000, 'scorer_url': 'http://a',
                  'scorer_urls': ['http://a', 'http://b'], 'lr': 1e-6, 'max_tokens_per_gpu': 16384}
        command = ['python3', 'train.py', '--sglang-max-running-requests', '1024', '--sglang-server-concurrency', '1600',
                   '--sglang-cuda-graph-max-bs-decode', '1024', '--sglang-mem-fraction-static', '0.8',
                   '--use-distributed-post', '--num-rollout', '2']
        line = console.settings_line(config, command, {'MIXRL_KEEP_TRAIN_SAMPLES': '1'})
        for part in ('top-p 0.95, top-k 20', 'R3 replay: on', 'refill rounds: 2', 'oversample: 0.3',
                     'in flight: 3200 responses, 1400 grading, 100000 groups', 'reward services: 2',
                     'SGLang per engine: 1024 running, request cap 1600, CUDA graphs to 1024, memory 0.8',
                     'distributed post: on', 'keep train responses: 1', 'max tokens/GPU: 16384'):
            self.assertIn(part, line)
        line = console.settings_line(dict(config, routing_replay=0, scorer_urls=None), command[:2], {})
        for part in ('R3 replay: OFF', 'reward services: 1', 'auto running', 'distributed post: off'):
            self.assertIn(part, line)


class StepClockTests(unittest.TestCase):
    def test_phases_and_total(self):
        times = iter([100.0, 100.0, 700.0, 700.0, 940.0, 940.0, 960.0, 961.0])
        with patch('slime_plugins.chimera_mixrl.steplog.time.monotonic', lambda: next(times)), \
                patch('builtins.print') as printed:
            clock = StepClock(3, 400)
            with clock.phase('rollout'):
                pass
            with clock.phase('train'):
                pass
            with clock.phase('sync'):
                pass
            line = clock.emit()
        self.assertEqual(line, {'rollout_id': 3, 'num_rollout': 400, 'seconds': 861.0, 'rollout': 600.0,
                                'train': 240.0, 'sync': 20.0})
        self.assertTrue(printed.call_args[0][0].startswith('MIXRL_STEP {'))


if __name__ == '__main__':
    unittest.main()
