"""mixrl/run.sh, reward.sh and judge.sh: host checks and the exact docker/vLLM calls, with stubs."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import threading
import unittest

REPO = Path(__file__).resolve().parents[1]
MIXRL = REPO / 'mixrl'
STUB = '''#!/usr/bin/env bash
# Records each call under a lock (callers run some in the background); docker copies
# --env-file before the caller deletes it.
(
    flock 9
    printf '%s\\n' "$@" --- >&9
) 9>> "$STUB_LOG/$(basename "$0").calls"
if [[ "$(basename "$0")" == docker ]]; then
    prev=
    for arg in "$@"; do
        if [[ "$prev" == --env-file ]]; then cp "$arg" "$STUB_LOG/env_file"; fi
        prev=$arg
    done
    # `docker ps --format {{.Names}}` lists the running reward-service containers; other `ps` calls only test for
    # a match. STUB_NO_CONTAINERS=1: every started container has already died.
    if [[ "$1" == ps && -n "${STUB_NO_CONTAINERS:-}" ]]; then :
    elif [[ "$1" == ps && " $* " == *"{{.Names}}"* ]]; then echo mixrl-reward-service
    elif [[ "$1" == ps ]]; then echo running; fi
    if [[ "$1" == inspect ]]; then echo "${STUB_RESTARTS:-0}"; fi
    # A real `docker run` lasts the whole run; give background log followers time to start.
    if [[ "$1" == run && " $* " == *" --gpus "* ]]; then sleep 0.5; fi
fi
exit 0
'''


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


class Services(BaseHTTPRequestHandler):
    def do_GET(self):
        body = {'/health': {'protocol_id': 'x', 'judge': {'model': 'mixrl-judge', 'ready': True}, 'task_errors': {},
                            'task_judge': {}},
                '/v1/models': {'data': [{'id': 'mixrl-judge'}]}}.get(self.path)
        self.send_response(200 if body else 404)
        self.end_headers()
        self.wfile.write(json.dumps(body or {}).encode())

    def log_message(self, *args):
        pass


class ScriptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.base = root / 'base'
        for path in ('models/m/hf/config.json', 'models/m/mcore/latest_checkpointed_iteration.txt',
                     'datasets/d/manifest.json', 'repos/transformers/src/transformers/models/chimera/__init__.py',
                     'repos/chimera-eval/eval_stack/reward_service.py'):
            (self.base / path).parent.mkdir(parents=True, exist_ok=True)
            (self.base / path).write_text('{}')
        (self.base / 'models/judge').mkdir()
        self.log = root / 'log'
        self.log.mkdir()
        bin_dir = root / 'bin'
        bin_dir.mkdir()
        for name in ('docker', 'vllm'):
            (bin_dir / name).write_text(STUB)
            (bin_dir / name).chmod(0o755)
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Services)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        port = str(self.server.server_port)
        self.env = dict(os.environ, PATH=f'{bin_dir}:{os.environ["PATH"]}', STUB_LOG=str(self.log),
                        BASE_DIR=str(self.base), MODEL_NAME='m', DATASET_NAME='d', JUDGE_MODEL_DIR='judge',
                        REWARD_PORT=port, JUDGE_PORT=port)

    def run_script(self, *args, **env):
        return subprocess.run(['bash', str(MIXRL / args[0]), *args[1:]], env=dict(self.env, **env),
                              capture_output=True, text=True, timeout=30)

    def calls(self, name):
        path = self.log / f'{name}.calls'
        return [c.strip('\n').split('\n') for c in path.read_text().split('---\n') if c.strip()] if path.exists() else []

    def env_file(self):
        return dict(line.split('=', 1) for line in (self.log / 'env_file').read_text().splitlines())

    def test_tasks_preview_needs_no_docker(self):
        result = self.run_script('run.sh', 'tasks')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('16 of 16 tasks enabled; 1008 prompts per step x 16 responses = 16128 samples', result.stdout)
        self.assertEqual(self.calls('docker'), [])

    def test_preflight_uses_no_gpus_and_carries_every_setting(self):
        result = self.run_script('run.sh', 'preflight', 'check-1', LR='2e-6')
        self.assertEqual(result.returncode, 0, result.stderr)
        (call,) = self.calls('docker')
        self.assertEqual(call[:2], ['run', '--rm'])
        self.assertNotIn('--gpus', call)
        self.assertIn('DRY_RUN=1', call)
        self.assertIn('MIXRL_RUNS_ROOT=/tmp/mixrl-preflight', call)
        self.assertEqual(call[-3:], ['suryavikram6/slime:pinned', 'bash', 'mixrl/internal/launch.sh'])
        settings = self.env_file()
        names = [line.split('=')[0] for line in (MIXRL / 'config.env').read_text().splitlines()
                 if line[:1].isupper() and '=' in line]
        self.assertLessEqual(set(names), set(settings))
        self.assertEqual((settings['LR'], settings['RUN_NAME'], settings['RESUME'], settings['DATA_ROOT']),
                         ('2e-6', 'check-1', '0', '/data'))

    def test_start_and_resume(self):
        result = self.run_script('run.sh', 'start', 'gsm8k-01')
        self.assertEqual(result.returncode, 0, result.stderr)
        # Both services answer, so start reuses them, then runs the preflight (no GPUs) and the training.
        for line in ('tasks: 16 of 16 tasks enabled', 'judge: already running', 'reward service: already running',
                     'preflight: ok', 'training: start run gsm8k-01'):
            self.assertIn(line, result.stdout)
        calls = self.calls('docker')
        preflight, call = [c for c in calls if c[0] == 'run']
        self.assertIn('DRY_RUN=1', preflight)
        self.assertNotIn('--gpus', preflight)
        self.assertIn('"device=0,1,2,3,4,5"', call)
        self.assertIn(f'{self.base}/runs:/data/runs', call)
        self.assertNotIn('DRY_RUN=1', call)
        self.assertEqual(self.env_file()['RESUME'], '0')
        # The reward service's and judge's output for the run is followed into its logs folder.
        run_dir = self.base / 'runs/chimera/mixrl/gsm8k-01'
        followed = [c for c in calls if c[:2] == ['logs', '-f']]
        self.assertCountEqual([c[-1] for c in followed], ['mixrl-reward-service', 'mixrl-judge-server'])
        self.assertTrue((run_dir / 'logs/reward_service.log').exists())
        # An early failure leaves only logs and manifests: the name can be retried.
        (run_dir / 'manifests').mkdir(parents=True)
        (run_dir / 'manifests/mixrl_config.json').write_text('{}')
        (run_dir / 'logs/train.log').write_text('failed before the first rollout')
        self.assertEqual(self.run_script('run.sh', 'start', 'gsm8k-01').returncode, 0)
        (run_dir / 'rollouts/train-0').mkdir(parents=True)
        (run_dir / 'rollouts/train-0/metrics.json').write_text('{}')
        result = self.run_script('run.sh', 'start', 'gsm8k-01')
        self.assertIn('already exists', result.stderr)
        result = self.run_script('run.sh', 'resume', 'gsm8k-01')
        self.assertIn('no checkpoint to resume', result.stderr)
        (run_dir / 'checkpoints').mkdir()
        (run_dir / 'checkpoints/latest_checkpointed_iteration.txt').write_text('10')
        # A run that saved weights only cannot resume: said plainly, before any container.
        (run_dir / 'manifests/config.env').write_text('LR=1e-6\nNO_SAVE_OPTIM=1\n')
        result = self.run_script('run.sh', 'resume', 'gsm8k-01')
        self.assertEqual(result.returncode, 1)
        self.assertIn('saved weights only (NO_SAVE_OPTIM=1), so it cannot resume', result.stderr)
        (run_dir / 'manifests/config.env').write_text('LR=1e-6\nNO_SAVE_OPTIM=0\n')
        before = len([c for c in self.calls('docker') if c[0] == 'run'])
        result = self.run_script('run.sh', 'resume', 'gsm8k-01')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.env_file()['RESUME'], self.env_file()['RUN_NAME']), ('1', 'gsm8k-01'))
        self.assertEqual(len([c for c in self.calls('docker') if c[0] == 'run']), before + 1)  # no preflight

    def test_command_line_settings_reach_the_container_and_typos_are_reported(self):
        result = self.run_script('run.sh', 'start', 'run-c', LR='3e-6', MIXRL_PIPELINE_SECONDS='15',
                                 CHIMERA_ROUTING_REPLAY='0', MIXRL_EXTEND_CONSTANT_HORIZON='1',
                                 DATA_ROOT='/host/path', MIXRL_TASKS_CONFIG='/host/tasks.json', MIXRL_PIPLINE_SECONDS='5')
        self.assertEqual(result.returncode, 0, result.stderr)
        settings = self.env_file()
        # config.env settings and launcher switches alike; container paths stay the container's.
        self.assertEqual((settings['LR'], settings['MIXRL_PIPELINE_SECONDS'], settings['CHIMERA_ROUTING_REPLAY'],
                          settings['MIXRL_EXTEND_CONSTANT_HORIZON'], settings['DATA_ROOT']),
                         ('3e-6', '15', '0', '1', '/data'))
        self.assertNotIn('MIXRL_TASKS_CONFIG', settings)
        self.assertIn('warning: MIXRL_PIPLINE_SECONDS is set but is not a MixRL setting', result.stderr)
        self.assertNotIn('warning: MIXRL_PIPELINE_SECONDS', result.stderr)

    def test_start_brings_up_missing_services_and_stops_when_one_fails(self):
        # Judge down and needed by the enabled tasks: start runs judge.sh, whose container dies at once.
        result = self.run_script('run.sh', 'start', 'run-a', JUDGE_PORT=str(free_port()), STUB_NO_CONTAINERS='1')
        self.assertEqual(result.returncode, 1)
        self.assertIn('judge: starting on GPUs 6,7', result.stdout)
        self.assertIn('the judge did not start', result.stderr)
        runs = [c for c in self.calls('docker') if c[0] == 'run']
        self.assertEqual([c[c.index('--name') + 1] for c in runs], ['mixrl-judge-server'])
        # Judge up, reward service down: reward.sh runs; its container dies, so nothing trains.
        result = self.run_script('run.sh', 'start', 'run-b', REWARD_PORT=str(free_port()), STUB_NO_CONTAINERS='1')
        self.assertEqual(result.returncode, 1)
        self.assertIn('reward service: starting 1 process(es)', result.stdout)
        self.assertIn('the reward service did not start', result.stderr)
        runs = [c for c in self.calls('docker') if c[0] == 'run']
        self.assertFalse(any('mixrl/internal/launch.sh' in c for c in runs))  # neither preflight nor training
        log = self.base / 'runs/chimera/mixrl/run-b/logs/reward_start.log'
        self.assertIn('reward service did not become ready', log.read_text())

    def test_clear_errors_before_any_container(self):
        cases = (
            (('run.sh', 'start'), {'TRAIN_GPUS': '0'}, 'lists 1 GPUs but POLICY_GPUS=6'),
            (('run.sh', 'preflight'), {'REWARD_PORT': str(free_port())}, 'start it with mixrl/reward.sh'),
            (('run.sh', 'start'), {'MODEL_NAME': 'missing'}, 'check BASE_DIR, MODEL_NAME'),
            (('run.sh', 'start', 'bad/name'), {}, "use letters, digits"),
            (('judge.sh',), {'JUDGE_GPUS': '2,3', 'JUDGE_TP': '1', 'JUDGE_DP': '1'}, 'lists 2 GPUs but JUDGE_TP x JUDGE_DP = 1 x 1'),
            (('judge.sh',), {'JUDGE_CONTEXT': '65536'}, 'must not exceed JUDGE_MAX_MODEL_LEN'),
        )
        for args, env, message in cases:
            with self.subTest(args=args, env=env):
                result = self.run_script(*args, **env)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
        self.assertEqual(self.calls('docker'), [])
        self.assertEqual(self.run_script('run.sh', 'train').returncode, 2)

    def test_reward_service_start(self):
        result = self.run_script('reward.sh')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Judge is up', result.stdout)
        self.assertIn('Reward service ready. Judge mixrl-judge: reachable.', result.stdout)
        run = next(c for c in self.calls('docker') if c[0] == 'run')
        for expected in ('/var/run/docker.sock:/var/run/docker.sock', f'{self.base}/repos/chimera-eval:/opt/chimera-eval',
                         f'JUDGE_URL=http://127.0.0.1:{self.env["JUDGE_PORT"]}/v1', 'JUDGE_CONTEXT=32768',
                         'JUDGE_CHAT_TEMPLATE_KWARGS={"reasoning_strength":"low"}',
                         '--judge-revision', 'mixrl-judge', self.env['REWARD_PORT']):
            self.assertIn(expected, run)

    def test_reward_service_several_processes(self):
        # Two stub reward services on consecutive ports.
        while True:
            first = ThreadingHTTPServer(('127.0.0.1', 0), Services)
            try:
                second = ThreadingHTTPServer(('127.0.0.1', first.server_port + 1), Services)
                break
            except OSError:
                first.server_close()
        for server in (first, second):
            threading.Thread(target=server.serve_forever, daemon=True).start()
            self.addCleanup(server.server_close)
            self.addCleanup(server.shutdown)
        port = first.server_port
        result = self.run_script('reward.sh', REWARD_PROCESSES='2', REWARD_PORT=str(port), JUDGE_CONCURRENCY='256',
                                 REWARD_WORKERS='768', CODE_CONCURRENCY='16', JUDGE_KV_CACHE_NUM_TOKENS='800000',
                                 JUDGE_MAX_PENDING='1024')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Reward service ready (2 processes)', result.stdout)
        runs = [c for c in self.calls('docker') if c[0] == 'run']
        self.assertEqual(len(runs), 2)
        for i, run in enumerate(runs):
            self.assertEqual(run[run.index('--name') + 1], f'mixrl-reward-service-{i}')
            self.assertEqual(run[run.index('--port') + 1], str(port + i))
            self.assertIn(f'{self.base}/cache/scorer_cache/w{i}:/data/cache/scorer_cache', run)
            # Totals are split across the processes.
            for expected in ('JUDGE_CONCURRENCY=128', 'CODE_CONCURRENCY=8', 'JUDGE_KV_CACHE_NUM_TOKENS=400000',
                             'MAX_PENDING=512'):
                self.assertIn(expected, run)
            self.assertEqual(run[run.index('--workers') + 1], '384')

    def test_judge_replicas_and_speculative_switch(self):
        (self.base / 'models/judge-assistant').mkdir()
        result = self.run_script('judge.sh', JUDGE_USE_DOCKER='1', JUDGE_GPUS='2,3', JUDGE_TP='1', JUDGE_DP='2',
                                 JUDGE_SPECULATIVE='0', JUDGE_MAX_NUM_BATCHED_TOKENS='65536', JUDGE_API_SERVERS='4')
        self.assertEqual(result.returncode, 0, result.stderr)
        run = next(c for c in self.calls('docker') if c[0] == 'run')
        for flag, value in (('--tensor-parallel-size', '1'), ('--data-parallel-size', '2'),
                            ('--max-num-batched-tokens', '65536'), ('--api-server-count', '4')):
            self.assertEqual(run[run.index(flag) + 1], value)
        self.assertNotIn('--speculative-config', run)
        for flag, value in (('--kv-cache-dtype', 'fp8'), ('--performance-mode', 'throughput')):
            self.assertEqual(run[run.index(flag) + 1], value)
        for flag in ('--language-model-only', '--aggregate-engine-logging'):
            self.assertIn(flag, run)
        self.assertEqual(run[run.index('--max-num-seqs') + 1], '1024')  # vLLM 0.28 would stop at 512
        result = self.run_script('judge.sh', JUDGE_USE_DOCKER='1', JUDGE_GPUS='2,3', JUDGE_MAX_NUM_SEQS='0')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('--max-num-seqs', [c for c in self.calls('docker') if c[0] == 'run'][-1])  # 0: vLLM's own
        result = self.run_script('judge.sh', JUDGE_USE_DOCKER='1', JUDGE_GPUS='2,3', JUDGE_TP='2', JUDGE_DP='1',
                                 JUDGE_MAX_NUM_SEQS='512', JUDGE_SPECULATIVE='1')
        self.assertEqual(result.returncode, 0, result.stderr)
        run = [c for c in self.calls('docker') if c[0] == 'run'][-1]
        self.assertEqual(run[run.index('--max-num-seqs') + 1], '512')
        spec = json.loads(run[run.index('--speculative-config') + 1])
        self.assertEqual(spec['num_speculative_tokens'], 5)
        result = self.run_script('judge.sh', JUDGE_USE_DOCKER='1', JUDGE_GPUS='2,3', JUDGE_TP='1', JUDGE_DP='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('JUDGE_TP x JUDGE_DP = 1 x 1', result.stderr)

    def test_judge_native_and_docker(self):
        result = self.run_script('judge.sh', JUDGE_USE_DOCKER='0')
        self.assertEqual(result.returncode, 0, result.stderr)
        (call,) = self.calls('vllm')
        self.assertEqual(call[:2], ['serve', f'{self.base}/models/judge'])
        # Default layout: two replicas of the 30B judge, one per GPU, two front-end processes.
        for flag, value in (('--tensor-parallel-size', '1'), ('--data-parallel-size', '2'), ('--max-num-seqs', '1024'),
                            ('--api-server-count', '2'), ('--max-model-len', '32768'),
                            ('--served-model-name', 'mixrl-judge'), ('--host', '127.0.0.1')):
            self.assertEqual(call[call.index(flag) + 1], value)
        self.assertNotIn('--speculative-config', call)
        (self.base / 'models/judge-assistant').mkdir()
        result = self.run_script('judge.sh', JUDGE_USE_DOCKER='1', JUDGE_GPUS='2,3', JUDGE_TP='2', JUDGE_DP='1',
                                 JUDGE_SPECULATIVE='1')
        self.assertEqual(result.returncode, 0, result.stderr)
        run = next(c for c in self.calls('docker') if c[0] == 'run')
        self.assertEqual(run[run.index('--gpus') + 1], '"device=2,3"')
        self.assertTrue((self.base / 'cache/vllm_cache').is_dir())  # made by us, not by Docker as root
        self.assertIn('/models/judge', run)
        spec = json.loads(run[run.index('--speculative-config') + 1])
        self.assertEqual(spec['model'], '/models/judge-assistant')
        self.assertIn('Judge ready', result.stdout)

    def test_docker_judge_startup_crash_is_reported_not_left_looping(self):
        result = self.run_script('judge.sh', JUDGE_USE_DOCKER='1', JUDGE_PORT=str(free_port()), STUB_RESTARTS='2')
        self.assertEqual(result.returncode, 1)
        self.assertIn('the judge failed during startup', result.stderr)
        commands = [c[:2] for c in self.calls('docker')]
        self.assertIn(['logs', '--tail'], commands)
        self.assertEqual(commands[-1], ['rm', '-f'])


if __name__ == '__main__':
    unittest.main()
