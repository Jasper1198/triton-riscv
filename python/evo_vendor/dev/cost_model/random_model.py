from __future__ import annotations

import numpy as np
from typing import Callable, Optional, List, Any

import triton
from .cost_model import CostModel
from triton.runtime.autotuner import Config

class RandomCostModel(CostModel):
    def __init__(self, seed: Optional[int]):
        super().__init__()
        self._rng = np.random.default_rng(seed)

    def score(self, population: Optional[List[Any]] = None, extra_features=None) -> np.ndarray:
        assert population is not None, "lost population"

        n = len(population)
        if n == 0:
            return np.array([], dtype=np.float64)
        
        raw = self._rng.random(n)
        probs = raw / raw.sum()

        return probs
    
    def update(self, configs=None, latencies=None, extra_features=None):
        return 
    

        


