"""Opt-in synchronous boundary stop; never interrupt an optimizer/save operation."""
import os
import math
import time


class RunBudget:
    def __init__(self, seconds=0, reserve=1200, first_update=300, stop_file='', clock=time.monotonic):
        if (not all(math.isfinite(x) for x in (seconds, reserve, first_update))
                or seconds < 0 or reserve < 0 or first_update <= 0 or (seconds and reserve >= seconds)):
            raise ValueError('Invalid run time budget/reserve')
        self.seconds, self.reserve, self.estimate = seconds, reserve, first_update
        self.stop_file, self.clock = stop_file, clock
        self.started = clock()
        self.update_started = self.started

    @classmethod
    def from_env(cls):
        return cls(float(os.getenv('MIXRL_WALLCLOCK_SECONDS', '0')),
                   float(os.getenv('MIXRL_FINAL_RESERVE_SECONDS', '1200')),
                   float(os.getenv('MIXRL_INITIAL_UPDATE_SECONDS', '300')),
                   os.getenv('MIXRL_STOP_FILE', ''))

    def begin_update(self):
        self.update_started = self.clock()

    def finish_update(self):
        # Conservative running maximum, including rollout/judge time; not actor-only time.
        self.estimate = max(self.estimate, self.clock() - self.update_started)
        return self.should_stop()

    def should_stop(self):
        from pathlib import Path
        return bool((self.stop_file and Path(self.stop_file).exists()) or
                    (self.seconds and self.clock() - self.started + self.reserve + self.estimate >= self.seconds))
