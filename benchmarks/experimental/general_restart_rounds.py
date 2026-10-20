"""Independent fixed-budget Arnoldi rounds with a shared completed-set snapshot.

The graph contains a predetermined reset schedule, not host convergence decisions.
Only the first round uses the upstream supplied start. This is a diagnostic;
multiple starts do not certify that the requested spectrum has been discovered.
"""

from operator import index

import warp as wp

from benchmarks.upstream.run_benchmarks import StartedGeneral
from warpack.arnoldi import random_vector, scale_vector


class RestartRoundsGeneral(StartedGeneral):
    def __init__(self, *args, round_length=100, **kwargs):
        self.round_length = index(round_length)
        if self.round_length < 1:
            raise ValueError("round_length must be positive")
        super().__init__(*args, **kwargs)
        self.round_cycle = 0

    def initialize(self, seed=42):
        self.round_cycle = 0
        super().initialize(seed)

    def step(self):
        if self.round_cycle and self.round_cycle % self.round_length == 0:
            self.restart_index.fill_(-1)
            seed = 7919 + 4583 * (self.round_cycle // self.round_length)
            wp.launch(random_vector, self.n, [self.qrows[0], seed], device=self.device)
            self.length(0, self.norm)
            wp.launch(scale_vector, self.n, [self.qrows[0], self.norm], device=self.device)
            for j in range(1, self.keep):
                self.expand(j)
        super().step()
        self.round_cycle += 1

    def capture(self, max_iterations=100, adaptive=False, finalizer=None):
        if adaptive is not False:
            raise ValueError("Restart rounds experiment requires a fixed budget")
        return super().capture(max_iterations, adaptive=False, finalizer=finalizer)
