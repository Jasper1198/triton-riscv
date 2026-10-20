from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional, Any

class CostModel(ABC):
    def __init__(self, post_hook: Optional[Callable]=None):
       self.post_hook = post_hook
       
    
    @abstractmethod
    def score(self):
        ...
    
    @abstractmethod
    def update(self, configs=None, latencies=None):
        ...
    
    def run_post_hook(self, *args: Any, **kwargs: Any):
        if self.post_hook is not None:
            self.post_hook(*args, **kwargs)

