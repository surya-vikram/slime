"""Wall-clock time per training step, by phase, as one MIXRL_STEP line (read by mixrl/internal/console.py)."""
import json
import time
from contextlib import contextmanager


class StepClock:
    """Times the phases of one step (rollout, train, save, sync, eval) and prints them with the step total."""

    def __init__(self, rollout_id, num_rollout):
        self.rollout_id, self.num_rollout = rollout_id, num_rollout
        self.started, self.phases = time.monotonic(), {}

    @contextmanager
    def phase(self, name):
        started = time.monotonic()
        try:
            yield
        finally:
            self.phases[name] = self.phases.get(name, 0.0) + time.monotonic() - started

    def emit(self, **extra):
        line = {'rollout_id': self.rollout_id, 'num_rollout': self.num_rollout,
                'seconds': round(time.monotonic() - self.started, 1),
                **{name: round(seconds, 1) for name, seconds in self.phases.items()}, **extra}
        print('MIXRL_STEP ' + json.dumps(line), flush=True)
        return line


IMPORTED = time.monotonic()  # train() imports this module first thing


def ready():
    """Startup finished: models loaded and the first weights are in SGLang."""
    print('MIXRL_READY ' + json.dumps({'seconds': round(time.monotonic() - IMPORTED, 1)}), flush=True)
