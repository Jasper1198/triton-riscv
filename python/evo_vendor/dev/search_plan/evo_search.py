from __future__ import annotations

import math
import heapq
import random
from dataclasses import dataclass
import itertools
import numpy as np
import xgboost as xgb #xgboost-3.1.2
from typing import Dict, Tuple, List, Optional, Any

import triton
from dev.cost_model.cost_model import CostModel
from dev.cost_model.random_model import RandomCostModel
from dev.cost_model.xgb_model import XGBCostModel
from triton.runtime.autotuner import Config
from triton.runtime.jit import KernelInterface
from triton.runtime.driver import driver
from triton.runtime.errors import OutOfResources, PTXASError
from triton.compiler.errors import CompileTimeAssertionFailure

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# logging.basicConfig(level=logging.INFO)

@dataclass
class EvoParams:
    mutate_strategy: str = "exploration"   # exploration / exploitation / random

    def copy(self) -> "EvoParams":
        return EvoParams(
            mutate_strategy=self.mutate_strategy,
        )
    
STRATEGY_PARAMS = {
    "exploration": {
        "mutate_prob": 0.35,
        "std_frac": 0.25,
        "crossover_rate": 0.75,
        "swap_prob": 0.55,
    },
    "exploitation": {
        "mutate_prob": 0.12,
        "std_frac": 0.08,
        "crossover_rate": 0.85,
        "swap_prob": 0.35,
    },
    "random": {
        "mutate_prob": 0.70,
        "std_frac": 0.40,
        "crossover_rate": 0.20,
        "swap_prob": 0.80,
    },
}

class IntGene:
    def __init__(self, name: str, lo: int, hi: int, val: Optional[int] = None, is_metadata: bool = False, force_pow2: bool = False):
        assert lo <= hi
        self.is_metadata = is_metadata
        self.force_pow2 = force_pow2
        self.name = name
        self.lo = int(lo)
        self.hi = int(hi)
        self.val = int(val) if val is not None else self.lo
        self.clamp()

    @staticmethod
    def _next_pow2(x: int) -> int:
        """Smallest power of two >= x, for x>=1."""
        if x <= 1:
            return 1
        return 1 << (x - 1).bit_length()

    @staticmethod
    def _prev_pow2(x: int) -> int:
        """Largest power of two <= x, for x>=1."""
        if x <= 1:
            return 1
        return 1 << (x.bit_length() - 1)
    
    def clamp(self) -> None:
        if self.val < self.lo:
            self.val = self.lo
        elif self.val > self.hi:
            self.val = self.hi

        #if not metadata, force power-of-two within [lo, hi]
        if (not self.is_metadata) or self.force_pow2:
            # If range contains no power-of-two at all, just keep clamped value.
            if self.hi < 1:
                return

            v = max(1, int(self.val))
            cand = self._next_pow2(v)

            if cand > self.hi:
                # fallback to largest power-of-two <= hi
                cand = self._prev_pow2(self.hi)

            # If still below lo, try next power-of-two >= lo
            if cand < self.lo:
                cand = self._next_pow2(max(1, self.lo))
                if cand > self.hi:
                    # no valid power-of-two in [lo, hi]
                    return

            self.val = int(cand)

    def copy(self) -> IntGene:
        return IntGene(self.name, self.lo, self.hi, self.val, self.is_metadata, self.force_pow2)

    # # mutate globally
    # def randomize(self, rng: np.random.Generator) -> None:
    #     self.val = int(rng.integers(self.lo, self.hi + 1))

    # local mutation (slightly mutate within a given % of range)

    def randomize_int(self, rng: np.random.Generator) -> None:
        if self.lo == self.hi:
            self.val = self.lo
            return

        self.val = int(rng.integers(self.lo, self.hi + 1))
        self.clamp()


    def randomize_pow2(self, rng: np.random.Generator) -> None:
        vals = []
        x = 1
        while x < self.lo:
            x <<= 1
        while x <= self.hi:
            vals.append(x)
            x <<= 1

        if not vals:
            self.clamp()
            return

        self.val = int(rng.choice(vals))

    def mutate_gaussian(self, rng: np.random.Generator, std_frac: float = 0.15) -> None:
        if self.lo == self.hi:
            self.val = self.lo
            return

        sigma = max(1.0, (self.hi - self.lo) * std_frac)
        self.val = int(round(self.val + rng.normal(0.0, sigma)))
        self.clamp()

    def mutate_pow2_gaussian(self, rng: np.random.Generator, std_exp: float = 0.75):
        """
        Gaussian mutation in log2 space.

        """
        if self.is_metadata and not self.force_pow2:
            return
        
        if self.val <= 0:
            self.val = max(1, self.lo)

        # exponent
        exp = int(math.log2(self.val))

        # Gaussian mutation in exponent space
        new_exp = int(round(exp + rng.normal(0, std_exp)))

        # limit exponent range
        lo_exp = math.ceil(math.log2(max(1, self.lo)))
        hi_exp = math.floor(math.log2(self.hi))

        new_exp = max(lo_exp, min(hi_exp, new_exp))

        self.val = 1 << new_exp
    
    # def mutate(self, rng: np.random.Generator, std_frac: float = 0.15):
    #     if self.is_metadata:
    #         self.mutate_gaussian(rng, std_frac=std_frac)
    #     else:
    #         #TODO: decide whether to self-adapt 'std_exp'
    #         self.mutate_pow2_gaussian(rng)

    def mutate(
        self,
        rng: np.random.Generator,
        strategy: str = "exploration",
        std_frac: float = 0.15,
    ) -> None:
        if strategy == "random":
            if self.is_metadata and not self.force_pow2:
                self.randomize_int(rng)
            else:
                self.randomize_pow2(rng)
            return

        if strategy == "exploration":
            if self.is_metadata and not self.force_pow2:
                self.mutate_gaussian(rng, std_frac=max(std_frac, 0.20))
            else:
                self.mutate_pow2_gaussian(rng, std_exp=1.5)
            return

        if strategy == "exploitation":
            if self.is_metadata and not self.force_pow2:
                self.mutate_gaussian(rng, std_frac=min(std_frac, 0.08))
            else:
                self.mutate_pow2_gaussian(rng, std_exp=0.5)
            return

        raise ValueError(f"Unknown mutate strategy: {strategy}")

# One Config, One Chromosome
class Chromosome:
    def __init__(self, genes: List[IntGene], evo_params: Optional[EvoParams] = None):
        self.genes: List[IntGene] = genes
        self.evo_params = evo_params.copy() if evo_params is not None else EvoParams()

        # name -> gene for quick lookup
        self._by_name = {g.name: g for g in genes}
        assert len(self._by_name) == len(genes), "duplicate gene name"


    
    def get_gene_names(self) -> List[str]:
        return list(self._by_name.keys())
    
    def get_gene(self, name: str) -> IntGene:
        return self._by_name[name]
    
    # use this to avoid redundent latency test
    def signature(self):
        return tuple(sorted((g.name, g.val) for g in self.genes))
    

    
    @classmethod
    def from_config(cls, cfg: Config, bounds:Dict[str, Dict[str, Any]],evo_params: Optional[EvoParams] = None) -> Chromosome:
        genes = []
        params = cfg.all_kwargs()

        metadata = ['num_warps', 'num_stages', 'num_ctas']
        pow2_metadata = ["num_warps"]

        for name, val in params.items():

            is_metadata = name in metadata
            force_pow2 = name in pow2_metadata

            genes.append(
                IntGene(
                    name, 
                    bounds[name]['lo'], 
                    bounds[name]['hi'], 
                    val, 
                    is_metadata=is_metadata,
                    force_pow2=force_pow2)
            )   

        return cls(genes, evo_params=evo_params)   

    def copy(self) -> Chromosome:
        return Chromosome(
            [g.copy() for g in self.genes],
            evo_params=self.evo_params,
        )

    def to_kwargs(self) -> Dict[str, int]:
        return {g.name: int(g.val) for g in self.genes if not g.is_metadata}

    def to_config(self) -> Config:
        num_warps=self._by_name['num_warps'].val
        num_stages=self._by_name['num_stages'].val
        num_ctas=self._by_name['num_ctas'].val
        # RISC-V tile-only: no num_threads gene
    
        return Config(
            kwargs=self.to_kwargs(),
            num_warps=num_warps,
            num_stages=num_stages,
            num_ctas=num_ctas,
            
        )

    # def mutate(self, rng: np.random.Generator, mutate_prob: float, std_frac: Optional[float] = None) -> None:
    #     if mutate_prob is None:
    #         mutate_prob = self.evo_params.mutate_prob

    #     if std_frac is None:
    #         std_frac = self.evo_params.std_frac

    #     assert 0.0 <= mutate_prob <= 1.0
    #     assert 0.0 <= std_frac <= 1.0

    #     for g in self.genes:
    #         if rng.random() < mutate_prob:
    #             g.mutate(rng, std_frac=std_frac)

    def mutate(
        self,
        rng: np.random.Generator,
        strategy: Optional[str] = None,
    ) -> None:
        if strategy is None:
            strategy = self.evo_params.mutate_strategy

        params = STRATEGY_PARAMS[strategy]
        mutate_prob = params["mutate_prob"]
        std_frac = params["std_frac"]

        for g in self.genes:
            if rng.random() < mutate_prob:
                g.mutate(rng, strategy=strategy, std_frac=std_frac)
  



class EvoAutotuner(KernelInterface):
    def __init__(self, fn, arg_names, configs, key, pop_size=400, max_generation=10, max_tuned_configs=500, top_k=1, cost_model="", seed=0, do_bench=None, warmup_configs=200):
        """
        :param fn: kernel to be evaluated on hardware
        :param configs: a list of `Config`
        """
        self.fn = fn        
        self.seed = seed
        self.keys = key
        self.top_k = top_k
        self.grid = None
        self.cache: Dict[Tuple, Tuple[Config, float]] = {} # {config key: (Config, timing)} 
        self.bounds: Optional[Dict[str, Dict[str, Any]]] = None
        self.eval_fn = None #
        self.arg_names = arg_names
        self.pop_probs: Optional['np.ndarray'] = None
        self.population = None
        self.init_pop_cfgs: Optional[List[Config]] = None
        self.init_pop_chroms: Optional[List[Chromosome]] = None
        self._initialized: bool = False
        self.warmup_configs = warmup_configs
        self.warmup_pop_cfgs = None
        # self.best_config: Optional[Config] = None
        # self.best_timing: float = float("inf")

        if do_bench is None:
            self.do_bench = driver.active.get_benchmarker()
        else:
            self.do_bench = do_bench

        # genetic algorithm params
        self.use_lhs_warmup = True
        self.rng = np.random.default_rng(seed)
        self.pop_size = pop_size
        # self.mutate_std = 0
        self.mutate_prob = 0.2  # 0.5 # 0.15
        self.crossover_rate = 0.8 # 0.8
        self.swap_prob = 0.5
        self.max_generation = max_generation
        self.top_k_seen = set() # avoid duplicates
        self.top_k_heap: List[Tuple[float, Tuple, Chromosome]] = [] # (score, sig, chromosome)
        self.frontier_size = max(self.top_k * 5, self.top_k)
        self.frontier_seen = set()
        self.frontier_heap: List[Tuple[float, Tuple, Chromosome]] = []
        self.tested = set()

        # monitor the topk trend
        self.round_history = []

        # the total number of configs that will be benched on real mechine
        # the time budget
        self.max_tuned_configs = max_tuned_configs

        if not configs:
            self.configs = [Config({}, num_warps=4, num_stages=3, num_ctas=1)]
        else:
            # all config has to have the same kwargs
            # simple check on config args
            arg_val = configs[0].all_kwargs()
            arg_size = len(arg_val)

            for cf in configs:
                assert arg_size == len(cf.all_kwargs()), "all configs should have the same total number of args"
                for k, _ in cf.all_kwargs().items():
                    assert k in arg_val, "all configs should have the same args"
                             
            self.configs = configs

        self.feature_order = self._build_feature_order()
        print("===== FEATURE ORDER =====")
        for i, name in enumerate(self.feature_order):
            print(f"[{i}] {name}")

        #TODO: add init booster
        if isinstance(cost_model, str):
            if cost_model in ("xgboost", "xgb"):
                self.cost_model = XGBCostModel(self.feature_order, seed)
            elif cost_model == "random":
                self.cost_model = RandomCostModel(seed)
            else:
                raise ValueError(f"Unknown cost model: {cost_model}")
            
    def _format_config_brief(self, cfg: Config) -> str:
        return (
            f"kwargs={cfg.kwargs}, "
            f"num_warps={cfg.num_warps}, "
            f"num_ctas={cfg.num_ctas}, "
            f"num_stages={cfg.num_stages}, "
            ""
        )

    def _dump_config_grid(self, cfg: Config, *args, **kwargs) -> Dict[str, float]:
        gf = self._grid_features(
            cfg,
            *args,
            **kwargs,
        )

        print(
            "[GRID] "
            f"grid=({int(gf['grid_0'])}, {int(gf['grid_1'])}, {int(gf['grid_2'])}) | "
            f"{self._format_config_brief(cfg)}"
        )

        return gf

    def _resolve_grid_tuple(self, config: Config, *args, **kwargs):
        """
        Resolve Triton launch grid into a normalized 3D tuple.

        Supports:
        - grid = (x,)
        - grid = (x, y)
        - grid = (x, y, z)
        - grid = lambda META: (...)
        """

        grid = None
        if callable(self.grid):
            # Triton JITFunction.run() does:
            #     bound_args, specialization, options = binder(*args, **kwargs)
            #     if callable(grid):
            #         grid = grid(bound_args)
            #
            # Here we approximate bound_args using arg_names + kwargs + config meta.
            meta = dict(zip(self.arg_names, args))
            meta.update(kwargs)
            meta.update(config.all_kwargs())

            grid = self.grid(meta)
        else:
            grid = self.grid

        if not isinstance(grid, tuple):
            grid = (grid,)

        grid_size = len(grid)
        grid_0 = int(grid[0]) if grid_size > 0 else 1
        grid_1 = int(grid[1]) if grid_size > 1 else 1
        grid_2 = int(grid[2]) if grid_size > 2 else 1

        return (grid_0, grid_1, grid_2)

    # every config needs to compute their own grid (grid can be callable object)
    def _grid_features(self, config: Config, *args, **kwargs) -> Dict[str, float]:
        grid_0, grid_1, grid_2 = self._resolve_grid_tuple(
            config,
            *args,
            **kwargs,
        )

        return {
            "grid_0": float(grid_0),
            "grid_1": float(grid_1),
            "grid_2": float(grid_2),
        }

    def _population_grid_features(self, population: List[Chromosome], *args, **kwargs):
        features = []

        for chrom in population:
            cfg = chrom.to_config()
            features.append(
                self._grid_features(
                    cfg,
                    *args,
                    **kwargs,
                )
            )

        return features


    def _configs_grid_features(self, configs: List[Config], *args, **kwargs):
        features = []

        for cfg in configs:
            features.append(
                self._grid_features(
                    cfg,
                    *args,
                    **kwargs,
                )
            )

        return features


    def _config_to_signature(self, cfg: Config):
        chrom = Chromosome.from_config(cfg, self.bounds)
        return chrom.signature()


    def _build_feature_order(self):
        names = set()
        cfg = self.configs[0]
        names.update(cfg.all_kwargs().keys())
        names.update(["grid_0", "grid_1", "grid_2"])
        
        return sorted(names)
    
    def _dump_tuning_history(self):
        if not self.round_history:
            print("\n===== TUNING HISTORY =====")
            print("No tuning history recorded.\n")
            return

        print("\n===== TUNING HISTORY =====")
        print("round | tested | round_best_ms | global_best_ms | improved")
        print("------|--------|---------------|----------------|---------")

        for h in self.round_history:
            print(
                f"{h['round']:>5} | "
                f"{h['num_tested']:>6} | "
                f"{h['round_best_latency']:>13.3f} | "
                f"{h['global_best_latency']:>14.3f} | "
                f"{str(h['improved']):>8}"
            )
        print()

    def _dump_population(self, generation: int, population: List[Chromosome], scores) -> None:
        print(f"\n===== GENERATION {generation} POPULATION =====")
        print(f"population size: {len(population)}")

        for i, (chrom, score) in enumerate(zip(population, scores)):
            cfg = chrom.to_config()
            print(
                f"[{i}] score={float(score):.6f} | "
                f"kwargs={cfg.kwargs}, "
                f"num_warps={cfg.num_warps}, "
                f"num_ctas={cfg.num_ctas}, "
                f"num_stages={cfg.num_stages}, "
                ""
            )    
    
    def _dump_topk(self, generation: int) -> None:
        print(f"\n===== GENERATION {generation} TOP-K =====")
        print(f"top_k_heap size: {len(self.top_k_heap)}")

        # heap 是 min-heap，這裡倒序印，best first
        items = sorted(self.top_k_heap, key=lambda x: x[0], reverse=True)

        for rank, (score, sig, chrom) in enumerate(items):
            cfg = chrom.to_config()
            print(
                f"[rank {rank}] score={float(score):.6f} | sig={sig} | "
                f"kwargs={cfg.kwargs}, "
                f"num_warps={cfg.num_warps}, "
                f"num_ctas={cfg.num_ctas}, "
                f"num_stages={cfg.num_stages}, "
                ""
            )

    def _is_gpu_backend(self):
        try:
            from triton.runtime.driver import driver
            target = driver.active.get_current_target()
            return target.backend in ("cuda", "hip")
        except Exception:
            return False

    def _bench(self, config, *args, **meta):
        # check for conflicts, i.e. meta-parameters both provided
        # as kwargs and by the autotuner
        conflicts = meta.keys() & config.kwargs.keys()
        if conflicts:
            raise ValueError(f"Conflicting meta-parameters: {', '.join(conflicts)}."
                             " Make sure that you don't re-define auto-tuned symbols.")
        

        current = dict(meta, **config.all_kwargs())
        if self._is_gpu_backend():
            current.pop("num_threads", None)

        def kernel_call():
            try:
                self.fn.run(
                    *args,
                    **current,
                )
            except Exception as e:
                raise
        
        try:
            times = self.do_bench(kernel_call, rep=10, quantiles=(0.5, 0.2, 0.8))
            return times[0]
        except (OutOfResources, CompileTimeAssertionFailure, PTXASError) as e:
            print(f"Autotuning failed with {e}")
            return float("inf")
    
    # TODO: test latencies for top k configs, also record the best
    # also update the cost model
    def test_top_k_latency(self, top_k_configs, *args, **kwargs) -> Tuple[Config, float]:
        timings = {}
        latencies = []
        
        for cfg in top_k_configs:
            try:
                self._dump_config_grid(cfg, *args, **kwargs)
                t = self._bench(cfg, *args, **kwargs)
                timings[cfg] = t
            except Exception as e:
                print(f"config failed: {cfg} -> {e}")
                timings[cfg] = float("inf")
        
        valid_cfgs = []
        valid_latencies = []

        for cfg, t in timings.items():
            if not np.isinf(t) and not np.isnan(t):
                valid_cfgs.append(cfg)
                valid_latencies.append(float(t))

        if valid_cfgs:
            extra_features = self._configs_grid_features(
                valid_cfgs,
                *args,
                **kwargs,
            )

            self.cost_model.update(
                valid_cfgs,
                valid_latencies,
                extra_features=extra_features,
            )
        else:
            print("No valid configs in this round; skip cost model update.")

        # latencies = [timings[cfg] for cfg in top_k_configs]

        # extra_features = self._configs_grid_features(
        #     top_k_configs,
        #     *args,
        #     **kwargs,
        # )

        # self.cost_model.update(
        #     top_k_configs,
        #     latencies,
        #     extra_features=extra_features,
        # )

        best = min(timings, key=timings.get)
        print("\n===== BENCH RESULT =====")
        for cfg, t in timings.items():
            print(f"{t:.3f} ms  |  {cfg}")

        print(f"\nBEST = {timings[best]:.3f} ms  |  {best}\n")
    

        return (best, timings[best])

    def _reset_search_state(self, reset_cost_model: bool = False):
        self.tested = set()
        self.top_k_heap = []
        self.top_k_seen = set()
        self.frontier_heap = []
        self.frontier_seen = set()
        self.grid = None

        self.init_pop_cfgs = None
        self.init_pop_chroms = None
        self.warmup_pop_cfgs = None

        self.population = None
        self.pop_probs = None
        self.bounds = None
        self._initialized = False

        self.round_history = []

        if reset_cost_model:
            if isinstance(self.cost_model, XGBCostModel):
                self.cost_model = XGBCostModel(self.feature_order, self.seed)
            elif isinstance(self.cost_model, RandomCostModel):
                self.cost_model = RandomCostModel(self.seed)    

    def _reserve_untested_configs(self, configs: List[Config]) -> List[Config]:
        """
        Remove configs already in self.tested, and immediately reserve the remaining
        ones by adding them into self.tested before benchmarking.
        """
        new_configs = []

        for cfg in configs:
            sig = self._config_to_signature(cfg)
            if sig in self.tested:
                continue
            self.tested.add(sig)
            new_configs.append(cfg)

        return new_configs

    def _update_topk(self, chromosomes: List[Chromosome], scores):
        """
        Maintain top-K best chromosomes.

        heap stores (score, signature, chromosome)
        smallest score at heap[0]
        """

        assert len(chromosomes) == len(scores), "lengths of chromosome and score are not equal"
        for chrom, score in zip(chromosomes, scores):
            sig = chrom.signature()

            # skip duplicates
            if sig in self.top_k_seen:
                continue
            if sig in self.tested:
                continue

            item = (score, sig, chrom.copy())

            if len(self.top_k_heap) < self.top_k:
                heapq.heappush(self.top_k_heap, item)
                self.top_k_seen.add(sig)

            else:
                # heap[0] = worst in top-K
                if score > self.top_k_heap[0][0]:
                    _, old_sig, _ = heapq.heapreplace(self.top_k_heap, item)
                    self.top_k_seen.discard(old_sig)
                    self.top_k_seen.add(sig)
  
    def _update_frontier(self, chromosomes: List[Chromosome], scores):
        assert len(chromosomes) == len(scores)

        for chrom, score in zip(chromosomes, scores):
            sig = chrom.signature()

            if sig in self.frontier_seen:
                continue
            if sig in self.tested:
                continue

            item = (score, sig, chrom.copy())

            if len(self.frontier_heap) < self.frontier_size:
                heapq.heappush(self.frontier_heap, item)
                self.frontier_seen.add(sig)
            else:
                if score > self.frontier_heap[0][0]:
                    _, old_sig, _ = heapq.heapreplace(self.frontier_heap, item)
                    self.frontier_seen.discard(old_sig)
                    self.frontier_seen.add(sig)

    # generate all int [lo, hi]
    def _all_ints(self, lo, hi):
        lo, hi = int(lo), int(hi)
        if lo > hi:
            return []   
        return list(range(lo, hi + 1))

    # generate points between lo hi with same distance
    def _linspace_int(self, lo, hi, num):
        lo, hi = int(lo), int(hi)
        if num <= 1 or lo == hi:
            return [lo]
        
        xs = [lo + (hi - lo) * i / (num - 1) for i in range(num)]
        xs = [int(round(x)) for x in xs]
        # sort and delete repeat
        xs = sorted(set(xs))
        return xs
    
    def _powers_of_two(self, lo, hi):
        lo, hi = int(lo), int(hi)
        if lo > hi:
            return []

        vals = []
        x = 1

        while x < lo:
            x *= 2
        
        while x <= hi:
            vals.append(x)
            x *= 2

        return vals
    
    # def _warmup_train_cost_model(self, *args, **kwargs):
    #     """
    #     Run a warmup phase before genetic evolution:
    #     benchmark a subset of initial LHS configs and use them to bootstrap the cost model.

    #     Returns
    #     -------
    #     int
    #         Number of configs actually benched in warmup.
    #     """
    #     assert self.init_pop_cfgs is not None, "Initial population configs are not ready"

    #     if self.warmup_configs <= 0:
    #         return 0

    #     n = min(self.warmup_configs, len(self.init_pop_cfgs))

    #     # Because initialize_population() already uses LHS,
    #     # taking the first n configs is already a reasonable LHS warmup subset.
    #     idxs = self.rng.choice(len(self.init_pop_cfgs), size=n, replace=False)
    #     warmup_cfgs = [self.init_pop_cfgs[i] for i in idxs]

    #     # Remove tested configs and reserve the new ones immediately
    #     warmup_cfgs = self._reserve_untested_configs(warmup_cfgs)

    #     if not warmup_cfgs:
    #         print("===== WARMUP =====")
    #         print("No new warmup configs available.\n")
    #         return 0

    #     print("===== WARMUP =====")
    #     print(f"Benchmarking {len(warmup_cfgs)} warmup configs...\n")

    #     timings: Dict[Config, float] = {}
    #     best_cfg = None
    #     best_lat = float("inf")

    #     for cfg in warmup_cfgs:
    #         try:
    #             t = self._bench(cfg, *args, **kwargs)
    #             timings[cfg] = t
    #             if not (np.isinf(t) or np.isnan(t)) and t < best_lat:
    #                 best_lat = t
    #                 best_cfg = cfg
    #         except Exception as e:
    #             print(f"warmup config failed: {cfg} -> {e}")
    #             timings[cfg] = float("inf")

    #     valid_cfgs: List[Config] = []
    #     valid_lats: List[float] = []

    #     print("===== WARMUP BENCH RESULT =====")
    #     for cfg, t in timings.items():
    #         print(f"{t:.3f} ms  |  {cfg}")
    #         if not (np.isinf(t) or np.isnan(t)):
    #             valid_cfgs.append(cfg)
    #             valid_lats.append(float(t))

    #     if valid_cfgs:
    #         self.cost_model.update(valid_cfgs, valid_lats)
    #         print(f"\nWarmup updated cost model with {len(valid_cfgs)} valid samples.\n")
    #     else:
    #         print("\nNo valid warmup samples collected.\n")

    #     return len(valid_cfgs), best_cfg, best_lat

    def _warmup_train_cost_model(self, *args, **kwargs):
        """
        Run a warmup phase before genetic evolution:
        1. Always include user-provided seed configs first
        2. Fill remaining slots with sampled LHS configs
        3. Benchmark them and bootstrap the cost model

        Returns
        -------
        Tuple[int, Optional[Config], float]
            (number_of_valid_samples, best_cfg, best_latency)
        """
        assert self.warmup_pop_cfgs is not None, "Warmup configs are not ready"

        if self.warmup_configs <= 0:
            return 0, None, float("inf")

        target_n = self.warmup_configs

        seed_cfgs = list(self.configs)

        remain = max(0, target_n - len(seed_cfgs))

        lhs_cfgs = self.warmup_pop_cfgs[:remain]
        if remain > 0 and len(self.warmup_pop_cfgs) > 0:
            lhs_cfgs = self.warmup_pop_cfgs[:remain]

            # avoid sampling duplicates from init_pop_cfgs 
            # idxs = self.rng.choice(len(self.init_pop_cfgs), size=min(remain, len(self.init_pop_cfgs)), replace=False)
            # lhs_cfgs = [self.init_pop_cfgs[i] for i in idxs]

        candidate_cfgs = seed_cfgs + lhs_cfgs

        seen = set()
        dedup_cfgs = []
        for cfg in candidate_cfgs:
            sig = self._config_to_signature(cfg)
            if sig in seen:
                continue
            seen.add(sig)
            dedup_cfgs.append(cfg)

        warmup_cfgs = dedup_cfgs[:target_n]

        warmup_cfgs = self._reserve_untested_configs(warmup_cfgs)

        if not warmup_cfgs:
            print("===== WARMUP =====")
            print("No new warmup configs available.\n")
            return 0, None, float("inf")

        print("===== WARMUP =====")
        print(f"Benchmarking {len(warmup_cfgs)} warmup configs...\n")

        timings: Dict[Config, float] = {}
        best_cfg = None
        best_lat = float("inf")

        for cfg in warmup_cfgs:
            try:
                t = self._bench(cfg, *args, **kwargs)
                timings[cfg] = t
                # if not (np.isinf(t) or np.isnan(t)) and t < best_lat:
                #     best_lat = t
                #     best_cfg = cfg
            except Exception as e:
                print(f"warmup config failed: {cfg} -> {e}")
                timings[cfg] = float("inf")

        best_cfg = min(timings, key=timings.get)
        best_lat = timings[best_cfg]
        
        valid_cfgs: List[Config] = []
        valid_lats: List[float] = []

        print("===== WARMUP BENCH RESULT =====")
        for cfg, t in timings.items():
            print(f"{t:.3f} ms  |  {cfg}")
            if not (np.isinf(t) or np.isnan(t)):
                valid_cfgs.append(cfg)
                valid_lats.append(float(t))

        if valid_cfgs:
            extra_features = self._configs_grid_features(
                valid_cfgs,
                *args,
                **kwargs,
            )

            self.cost_model.update(
                valid_cfgs,
                valid_lats,
                extra_features=extra_features,
            )

            print(f"\nWarmup updated cost model with {len(valid_cfgs)} valid samples.\n")
        else:
            print("\nNo valid warmup samples collected.\n")

        return len(valid_cfgs), best_cfg, best_lat

    def _generate_configs_lhs(
        self,
        bounds: Dict[str, Dict[str, Any]],
        *,
        fixed_config_option: Optional[Dict[str, int]] = None,
        max_generated_config: Optional[int] = None,
        metadata_points_per_key: int = 5, # deprecated
    ) -> List[triton.Config]:
        """
        Generate a list of triton.Config using Latin Hypercube Sampling (LHS).

        Design:
        - For metadata dims (num_warps / num_ctas / num_stages / num_threads / maxnreg),
        candidate values are generated by linspace integers.
        - For non-metadata dims, candidate values are powers of two within [lo, hi].
        - LHS is first performed in [0,1) for each dimension, then mapped to the
        discrete candidate list of that dimension.

        Parameters
        ----------
        bounds : Dict[str, Dict[str, Any]]
            Example:
                {
                    "BLOCK_SIZE_M": {"lo": 4, "hi": 64},
                    "BLOCK_SIZE_N": {"lo": 4, "hi": 64},
                    ...
                }

        fixed_config_option : Optional[Dict[str, int]]
            Force some keys to fixed values, e.g.
                {"num_threads": 8, "BLOCK_SIZE_K": 32}

        max_generated_config : Optional[int]
            Number of configs to generate. Defaults to self.pop_size.

        metadata_points_per_key : int
            For metadata parameters, how many discrete integer candidates to create
            between [lo, hi] before LHS mapping.

        Returns
        -------
        List[triton.Config]
        """
        keys = sorted(bounds.keys())

        if fixed_config_option is None:
            fixed_config_option = {}

        if max_generated_config is None:
            max_generated_config = self.pop_size

        rng = self.rng
        metadata_keys = {"num_warps", "num_ctas", "num_stages", "maxnreg"}
        pow2_metadata_keys = {"num_warps"}

        # ---------------------------------------------------------
        # 1. Build discrete candidate values for each dimension
        # ---------------------------------------------------------
        values_per_key: Dict[str, List[int]] = {}

        for k in keys:
            if k in fixed_config_option:
                values_per_key[k] = [int(fixed_config_option[k])]
                continue

            lo = int(bounds[k]["lo"])
            hi = int(bounds[k]["hi"])

            if k in pow2_metadata_keys:
                vals = self._powers_of_two(lo, hi)
            elif k in metadata_keys:
                # vals = self._linspace_int(lo, hi, metadata_points_per_key)
                vals = self._all_ints(lo, hi)
            else:
                vals = self._powers_of_two(lo, hi)
                if not vals:
                    raise ValueError(
                        f"No power-of-two value exists in range [{lo}, {hi}] for parameter '{k}'"
                    )

            if not vals:
                raise ValueError(f"No candidate values generated for key '{k}'")

            values_per_key[k] = sorted(set(int(v) for v in vals))

        # ---------------------------------------------------------
        # 2. Config constructor helper
        # ---------------------------------------------------------
        def make_config(kwargs: Dict[str, int]) -> triton.Config:
            kwargs = dict(kwargs)

            default = Config({})

            num_warps = kwargs.pop("num_warps", default.num_warps)
            num_ctas = kwargs.pop("num_ctas", default.num_ctas)
            num_stages = kwargs.pop("num_stages", default.num_stages)
            # RISC-V tile-only: no num_threads option

            return triton.Config(
                kwargs,
                num_warps=num_warps,
                num_ctas=num_ctas,
                num_stages=num_stages,
                
            )

        # ---------------------------------------------------------
        # 3. If total space is small, enumerate all
        # ---------------------------------------------------------
        total = 1
        for k in keys:
            total *= len(values_per_key[k])

        if total <= max_generated_config:
            generated: List[triton.Config] = []
            for combo in itertools.product(*(values_per_key[k] for k in keys)):
                kwargs = dict(zip(keys, combo))
                generated.append(make_config(kwargs))
            return generated

        # ---------------------------------------------------------
        # 4. Latin Hypercube Sampling in [0,1)
        #
        # For each dimension:
        #   - divide [0,1) into n strata
        #   - sample one point from each stratum
        #   - shuffle independently per dimension
        # Then map each sampled u to a discrete candidate bucket.
        # ---------------------------------------------------------
        n_samples = max_generated_config
        dim = len(keys)

        # lhs_matrix shape = [n_samples, dim]
        lhs_matrix = np.empty((n_samples, dim), dtype=np.float64)

        for j in range(dim):
            # one sample from each stratum
            cut_points = (np.arange(n_samples) + rng.random(n_samples)) / n_samples
            rng.shuffle(cut_points)
            lhs_matrix[:, j] = cut_points

        # ---------------------------------------------------------
        # 5. Map each continuous LHS coordinate to discrete value
        # ---------------------------------------------------------
        generated: List[triton.Config] = []
        seen = set()

        for i in range(n_samples):
            kwargs: Dict[str, int] = {}

            for j, k in enumerate(keys):
                vals = values_per_key[k]

                if len(vals) == 1:
                    kwargs[k] = vals[0]
                    continue

                u = lhs_matrix[i, j]
                idx = min(int(u * len(vals)), len(vals) - 1)
                kwargs[k] = vals[idx]

            cfg = make_config(kwargs)

            # deduplicate because discrete mapping can collapse different LHS points
            sig = (
                tuple(sorted(cfg.kwargs.items())),
                cfg.num_warps,
                cfg.num_ctas,
                cfg.num_stages,

            )
            if sig in seen:
                continue
            seen.add(sig)
            generated.append(cfg)

        # ---------------------------------------------------------
        # 6. If dedup made us short, fill remaining by random sampling
        # ---------------------------------------------------------
        max_attempts = max(50, (n_samples - len(generated)) * 20)
        attempts = 0

        while len(generated) < n_samples and attempts < max_attempts:
            attempts += 1
            kwargs = {}
            for k in keys:
                vals = values_per_key[k]
                kwargs[k] = int(rng.choice(vals))

            cfg = make_config(kwargs)
            sig = (
                tuple(sorted(cfg.kwargs.items())),
                cfg.num_warps,
                cfg.num_ctas,
                cfg.num_stages,

            )
            if sig in seen:
                continue
            seen.add(sig)
            generated.append(cfg)

        return generated

    def _generate_configs_from_bounds(
        self,
        bounds: Dict[str, Dict[str, Any]],
        *,
        points_per_key = 5,
        fixed_config_option: Optional[Dict[str, int]] = None,
        max_generated_config: int = None,
    ):
        """
        Generate a list of triton.Config from bounds.

        - If total grid size <= max_generated_config: enumerate all combinations (grid).
        - Else: randomly sample combinations (uniform over the discretized grid).
        - fixed_config_option: dict like {"BLOCK_SIZE_K": 32} to fix some keys.
        """
        keys = sorted(bounds.keys())

        if fixed_config_option is None:
            fixed_config_option = {}

        if max_generated_config is None:
            max_generated_config = self.pop_size
        
        rng = self.rng
        meta_data = ["num_warps", "num_ctas", "num_stages", "maxnreg"]
        pow2_meta_data = {"num_warps"}

        # every list inside value_per_key indicate 
        # the key's possible value when generating config 
        values_per_key: List[List[int]] = []
        for k in keys:
            if k in fixed_config_option:
                logger.debug(f"fixed config option [{k}]")
                values_per_key.append([fixed_config_option[k]])
                continue

            lo = bounds[k]["lo"]
            hi = bounds[k]["hi"]
            num = points_per_key
            
            if k in pow2_meta_data:
                vals = self._powers_of_two(lo, hi)
                values_per_key.append(vals)
            elif k in meta_data:
                vals = self._linspace_int(lo, hi, num)
                values_per_key.append(vals)
            else:
                vals = self._powers_of_two(lo, hi)
                if not vals:
                    raise ValueError(
                        f"No power-of-two value exists in range [{lo}, {hi}] for parameter '{k}'"
                    )
                values_per_key.append(vals)

        # the number of all possible configs
        total = 1
        for vals in values_per_key:
            total *= len(vals)
        
        logger.debug(f"there are {total} possible configs")

        # Helper to wrap kwargs -> triton.Config
        # Preserve num_threads/num_warps/etc from the seed config if present
        def make_config(kwargs: Dict[str, int]) -> triton.Config:
            kwargs = dict(kwargs)

            default = Config({})

            num_warps   = kwargs.pop("num_warps", default.num_warps)
            num_ctas    = kwargs.pop("num_ctas", default.num_ctas)
            num_stages  = kwargs.pop("num_stages", default.num_stages)
            # RISC-V tile-only: no num_threads option

            return triton.Config(
                kwargs,
                num_warps=num_warps,
                num_ctas=num_ctas,
                num_stages=num_stages,
                
            )

        generated: List[triton.Config] = []

        # generate configs if total does not exceed max_generated_config
        if total <= max_generated_config:
            for combo in itertools.product(*values_per_key):
                kwargs = dict(zip(keys, combo))
                generated.append(make_config(kwargs))
            return generated
        

        # Otherwise sample without enumerating the whole product
        # Sample unique combinations (avoid duplicates)
        seen = set()
        # Upper bound on attempts to avoid infinite loop when max_generated_config ~ total
        max_attempts = max_generated_config * 10

        attempts = 0
        while len(generated) < max_generated_config and attempts < max_attempts:
            attempts += 1
            combo = tuple(rng.choice(vals) for vals in values_per_key)
            if combo in seen:
                continue
            seen.add(combo)
            kwargs = dict(zip(keys, combo))
            logger.debug(f"generate: {kwargs}")
            generated.append(make_config(kwargs))
        
        return generated
    
    def _compute_bounds(self) -> Dict[str, Dict[str, int]]:
        keys = sorted(set().union(*(c.all_kwargs().keys() for c in self.configs)))
        out: Dict[str, Dict[str, int]] = {}
        for k in keys:
            vals = [
                c.all_kwargs()[k]
                for c in self.configs
                if k in c.all_kwargs() and isinstance(c.all_kwargs()[k], (int, float))
            ]
            if vals:
                out[k] = {"lo": min(vals), "hi": max(vals)}
            # else:
            #     out[k] = None  
        return out

    def _configs_to_chromosomes(self, cfgs: List[Config]) -> List[Chromosome]:
        assert self.bounds is not None
        return [Chromosome.from_config(cfg, self.bounds) for cfg in cfgs]

    def _sample_next_strategy(
        self,
        parent_strategy: str,
        child_better: bool,
        near_tie: bool = False,
    ) -> str:
        """
        based on child / parent scores to choose the next generation strategy
        """
        transitions = {
            "exploration": {
                "win":  {"exploration": 0.25, "exploitation": 0.65, "random": 0.10},
                "lose": {"exploration": 0.70, "exploitation": 0.15, "random": 0.15},
                "tie":  {"exploration": 0.50, "exploitation": 0.25, "random": 0.25},
            },
            "exploitation": {
                "win":  {"exploration": 0.10, "exploitation": 0.80, "random": 0.10},
                "lose": {"exploration": 0.60, "exploitation": 0.25, "random": 0.15},
                "tie":  {"exploration": 0.35, "exploitation": 0.50, "random": 0.15},
            },
            "random": {
                "win":  {"exploration": 0.25, "exploitation": 0.45, "random": 0.30},
                "lose": {"exploration": 0.45, "exploitation": 0.15, "random": 0.40},
                "tie":  {"exploration": 0.30, "exploitation": 0.20, "random": 0.50},
            },
        }

        if near_tie:
            mode = "tie"
        elif child_better:
            mode = "win"
        else:
            mode = "lose"

        probs_dict = transitions[parent_strategy][mode]
        names = list(probs_dict.keys())
        probs = [probs_dict[n] for n in names]
        return self.rng.choice(names, p=probs)

    def _family_competition(
        self,
        parents: List[Chromosome],
        children: List[Chromosome],
        parent_scores,
        child_scores,
        eps: float = 1e-6,
    ) -> List[Chromosome]:
        """
        choose survivors and strategy based on score
        """
        assert len(parents) == len(children)
        assert len(parent_scores) == len(child_scores)

        survivors = []

        for p, c, ps, cs in zip(parents, children, parent_scores, child_scores):
            near_tie = abs(cs - ps) <= eps

            if cs > ps + eps:
                survivor = c.copy()
                next_strategy = self._sample_next_strategy(
                    parent_strategy=p.evo_params.mutate_strategy,
                    child_better=True,
                    near_tie=False,
                )
            elif ps > cs + eps:
                survivor = p.copy()
                next_strategy = self._sample_next_strategy(
                    parent_strategy=p.evo_params.mutate_strategy,
                    child_better=False,
                    near_tie=False,
                )
            else:
                survivor = c.copy()
                next_strategy = self._sample_next_strategy(
                    parent_strategy=p.evo_params.mutate_strategy,
                    child_better=False,
                    near_tie=True,
                )

            survivor.evo_params.mutate_strategy = next_strategy
            survivors.append(survivor)

        return survivors
    
    def _pick_untested_from_heap(self, heap_items, k: int) -> List[Config]:
        items = sorted(heap_items, key=lambda x: x[0], reverse=True)

        out = []
        seen_local = set()

        for score, sig, chrom in items:
            if sig in self.tested:
                continue
            if sig in seen_local:
                continue
            out.append(chrom.to_config())
            seen_local.add(sig)
            if len(out) >= k:
                break

        return out

    def _propose_candidates(self, k: int) -> List[Config]:
        selected = []
        selected_sigs = set()

        def add_configs(cfgs: List[Config]):
            for cfg in cfgs:
                sig = self._config_to_signature(cfg)
                if sig in self.tested:
                    continue
                if sig in selected_sigs:
                    continue
                selected.append(cfg)
                selected_sigs.add(sig)
                if len(selected) >= k:
                    return True
            return False

        top_cfgs = self._pick_untested_from_heap(self.top_k_heap, k)
        if add_configs(top_cfgs):
            return selected

        frontier_cfgs = self._pick_untested_from_heap(self.frontier_heap, k)
        if add_configs(frontier_cfgs):
            return selected

        population_items = sorted(
            zip(self.pop_probs, self.population),
            key=lambda x: x[0],
            reverse=True,
        )
        pop_cfgs = [chrom.to_config() for score, chrom in population_items]
        add_configs(pop_cfgs)

        return selected

    def initialize_population(
        self,
        *,
        points_per_key=5,
        fixed_config_option: Optional[Dict[str, int]] = None,       
    ):
        # find upper bound and lower bound of the config options
        self.bounds = self._compute_bounds()

        # generate population
        all_cfgs = self._generate_configs_lhs(
            self.bounds,
            fixed_config_option=fixed_config_option,
            max_generated_config=self.pop_size + self.warmup_configs,
            metadata_points_per_key=points_per_key,
        )

        self.init_pop_cfgs = all_cfgs[:self.pop_size].copy()
        self.warmup_pop_cfgs = all_cfgs[self.pop_size:].copy()

        self.init_pop_chroms = self._configs_to_chromosomes(self.init_pop_cfgs)

        # genetic algo's use evolve_pop, beyond this scope, use Config instead
        self.population = self.init_pop_chroms.copy()
        self._initialized = True

        # infos
        print("\n===== INITIAL GENERATED CONFIGS =====")
        print(f"total generated configs: {len(all_cfgs)}")
        print(f"init cfgs: {len(self.init_pop_cfgs)}")
        print(f"warmup cfgs: {len(self.warmup_pop_cfgs)}")

        for i, cfg in enumerate(all_cfgs):
            print(f"[{i}] {cfg}")

    # Roulette Wheel Selection
    def select(self, num_select: int, elite_k: int = 0):
        assert self.population is not None, "population is None"
        assert self.pop_probs is not None, "pop_probs is None"
        assert 0 <= elite_k <= num_select
        
        n = len(self.population)
        probs = np.asarray(self.pop_probs, dtype=np.float64)

        if elite_k > 0:
            # find largest probs' idx (unsorted)
            elite_idx = np.argpartition(-probs, elite_k - 1)[:elite_k]
        else:
            elite_idx = np.array([], dtype=np.int64)
        
        rest = num_select - elite_k
        if rest > 0:
            roulette_idx = self.rng.choice(
                np.arange(n),
                size=rest,
                replace=True,
                p=probs
            )
            
        else:
            roulette_idx = np.array([], dtype=np.int64)

        # self.rng.shuffle(roulette_idx)

        elite = [self.population[i].copy() for i in elite_idx]
        parents = [self.population[i].copy() for i in roulette_idx]

        return elite, parents

    
    # def uniform_crossover(
    #     self,
    #     parents: List[Chromosome],
    #     *,
    #     crossover_rate: float = 0.9,
    #     swap_prob: float = 0.5,
    # ) -> List[Chromosome]:
    #     """
    #     Uniform crossover.
    #     - crossover_rate: 0.9 means 90% pairs will crossover.
    #     - swap_prob: per-gene swap probability (0.5 is standard).
    #     """
    #     assert 0.0 <= crossover_rate <= 1.0
    #     assert 0.0 <= swap_prob <= 1.0
    #     assert len(parents) >= 2

    #     children: List[Chromosome] = []

    #     # iterate parents pairly, if there's only one parent left, simply add it to children arr
    #     it = iter(parents)
    #     for p1 in it:
    #         try:
    #             p2 = next(it)
    #         except StopIteration:
    #             children.append(p1.copy())
    #             break

    #         # don't crossover
    #         if self.rng.random() >= crossover_rate:
    #             children.append(p1.copy())
    #             children.append(p2.copy())
    #             continue

    #         # crossover
    #         c1 = p1.copy()
    #         c2 = p2.copy()

    #         # align gene names
    #         if c1._by_name.keys() != c2._by_name.keys():
    #             raise ValueError("Parents have different gene sets; cannot crossover safely.")

    #         for name in c1.get_gene_names():
    #             g1 = c1.get_gene(name)
    #             g2 = c2.get_gene(name)

    #             # TODO: decide whether to optimize metadata
    #             # if g1.is_metadata:
    #             #     continue

    #             # per-gene swap
    #             if self.rng.random() < swap_prob:
    #                 g1.val, g2.val = g2.val, g1.val

    #         # for safety：clamp
    #         for g in c1.genes:
    #             g.clamp()
    #         for g in c2.genes:
    #             g.clamp()

    #         children.append(c1)
    #         children.append(c2)

    #     return children
    
    def uniform_crossover(
        self,
        parents: List[Chromosome],
        *,
        crossover_rate: Optional[float] = None,
        swap_prob: Optional[float] = None,
        self_adapt: bool = True,
    ) -> List[Chromosome]:
        """
        Uniform crossover.

        If self_adapt=True:
        - each child uses adapted evo_params from its parent.
        - crossover_rate and swap_prob come from the adapted params.

        If self_adapt=False:
        - use passed crossover_rate / swap_prob.
        """
        assert len(parents) >= 2

        children: List[Chromosome] = []

        it = iter(parents)
        for p1 in it:
            try:
                p2 = next(it)
            except StopIteration:
                child = p1.copy()
                children.append(child)
                break

            c1 = p1.copy()
            c2 = p2.copy()


            if self_adapt:
                p1_params = STRATEGY_PARAMS[p1.evo_params.mutate_strategy]
                p2_params = STRATEGY_PARAMS[p2.evo_params.mutate_strategy]

                cr = 0.5 * (p1_params["crossover_rate"] + p2_params["crossover_rate"])
                sp = 0.5 * (p1_params["swap_prob"] + p2_params["swap_prob"])
            else:
                cr = self.crossover_rate if crossover_rate is None else crossover_rate
                sp = self.swap_prob if swap_prob is None else swap_prob

            assert 0.0 <= cr <= 1.0
            assert 0.0 <= sp <= 1.0

            # don't crossover
            if self.rng.random() >= cr:
                children.append(c1)
                children.append(c2)
                continue

            # align gene names
            if c1._by_name.keys() != c2._by_name.keys():
                raise ValueError("Parents have different gene sets; cannot crossover safely.")

            for name in c1.get_gene_names():
                g1 = c1.get_gene(name)
                g2 = c2.get_gene(name)

                if self.rng.random() < sp:
                    g1.val, g2.val = g2.val, g1.val

            for g in c1.genes:
                g.clamp()
            for g in c2.genes:
                g.clamp()

            children.append(c1)
            children.append(c2)

        return children

    # TODO: do selection, mutation, crossover 
    def evolve(self, *args, **kwargs): 
        assert self.pop_probs is not None, "predict the population probs first"

        max_generation = self.max_generation
        num_generation = 0

        while num_generation < max_generation:
            # basic work flow of genetic algo
            reserved, parents = self.select(self.pop_size) # reserved is for elite, but now it is empty
            children = self.uniform_crossover(parents, self_adapt=True)

            for c in children:
                c.mutate(self.rng)

            # choose survivor and self-adapt tune strategy
            pair_n = min(len(parents), len(children))
            parents = parents[:pair_n]
            children = children[:pair_n]
            pair_population = parents + children
            pair_features = self._population_grid_features(pair_population, *args, **kwargs)
            pair_scores = self.cost_model.score(pair_population, extra_features=pair_features)
            parent_scores = pair_scores[:pair_n]
            child_scores = pair_scores[pair_n:]

            survivors = self._family_competition(
                parents,
                children,
                parent_scores,
                child_scores,
            )

            next_generation = reserved + survivors

            # update cost model
            extra_features = self._population_grid_features(
                next_generation,
                *args,
                **kwargs
            )
            scores = self.cost_model.score(next_generation, extra_features=extra_features)
            self.population = next_generation
            self.pop_probs = scores

            self._update_topk(next_generation, scores)
            self._update_frontier(next_generation, self.pop_probs)
            # self._dump_population(num_generation, next_generation, scores)
            # self._dump_topk(num_generation)

            num_generation += 1

        # # pick top k configs
        # top_k_configs = []
        # for _, _, chrom in self.top_k_heap:
        #     top_k_configs.append(chrom.to_config())

        return 
    
    
    
    # TODO:
    def run(self, *args, **kwargs):
        self.nargs = dict(zip(self.arg_names, args))
        used_cached_result = True
        if len(self.configs) > 1:
            all_args = {**self.nargs, **kwargs}
            _args = {k: v for (k, v) in all_args.items() if k in self.arg_names}
            key = [_args[key] for key in self.keys if key in _args]
            for _, arg in _args.items():
                if hasattr(arg, "dtype"):
                    key.append(str(arg.dtype))
            key = tuple(key)            
            # TODO: 
            reset_cost_model = True
            if key not in self.cache:
                self._reset_search_state(reset_cost_model)
                self.grid = kwargs.get("grid", None)
                used_cached_result = False
                
                max_tuned_configs = self.max_tuned_configs
                num_tuned_configs = 0

                # genetic algorithm 
                self.initialize_population()

                extra_features = self._population_grid_features(
                    self.population,
                    *args,
                    **kwargs,
                )
                self.pop_probs = self.cost_model.score(self.population, extra_features=extra_features)
                self.top_k_heap = []
                self.top_k_seen = set()
                self._update_topk(self.population, self.pop_probs)
                self._update_frontier(self.population, self.pop_probs)

                # LHS warmup before genetic algo
                if self.use_lhs_warmup and isinstance(self.cost_model, XGBCostModel):
                    warmed, warm_best_cfg, warm_best_lat = self._warmup_train_cost_model(*args, **kwargs)
                    num_tuned_configs += warmed

                    if warm_best_cfg is not None:
                        self.cache[key] = (warm_best_cfg, warm_best_lat)

                    extra_features = self._population_grid_features(
                        self.population,
                        *args,
                        **kwargs,
                    )
                    self.pop_probs = self.cost_model.score(self.population, extra_features=extra_features)
                    self.top_k_heap = []
                    self.top_k_seen = set()
                    self._update_topk(self.population, self.pop_probs)
                    self._update_frontier(self.population, self.pop_probs)

                while num_tuned_configs < max_tuned_configs:
                    self.evolve(*args, **kwargs)
                    proposal_cfgs = self._propose_candidates(self.top_k)
                    top_k_configs = self._reserve_untested_configs(proposal_cfgs)

                    if not top_k_configs:
                        print("no new top k configs to test in this round")
                        num_tuned_configs += self.top_k
                        continue

                    # (config, best_time_within_topk)
                    result = self.test_top_k_latency(top_k_configs, *args, **kwargs)

                    # round info
                    round_best_latency = result[1]
                    round_id = len(self.round_history)
                    improved = False

                    # update best config info
                    if key in self.cache:
                        # compare latency
                        if result[1] < self.cache[key][1]:
                            improved = True
                            self.cache[key] = result
                    else:
                        improved = True
                        self.cache[key] = result

                    self.round_history.append({
                        "round": round_id,
                        "num_tested": len(top_k_configs),
                        "round_best_latency": round_best_latency,
                        "global_best_latency": self.cache[key][1],
                        "improved": improved,
                        "round_best_config": result[0],
                        "global_best_config": self.cache[key][0],
                    })

                    extra_features = self._population_grid_features(
                        self.population,
                        *args,
                        **kwargs,
                    )
                    self.pop_probs = self.cost_model.score(self.population, extra_features=extra_features)
                    
                    # TBD: here erase the top k heap first
                    self.top_k_heap = []
                    self.top_k_seen = set()

                    num_tuned_configs += self.top_k
            config = self.cache[key][0]
        else:
            config = self.configs[0]

        self._dump_tuning_history()

        print(f"===== Tune Result =====")
        print(f"Config = {self.cache[key][1]:.3f} ms  |  {config}\n")

        current = dict(config.all_kwargs())
        if self._is_gpu_backend():
            current.pop("num_threads", None)

        # ret = self.fn.run(
        #     *args,
        #     **kwargs,
        #     **config.all_kwargs(),
        # )

        ret = self.fn.run(
            *args,
            **kwargs,
            **current,
        )

        self.nargs = None
        return ret


    
# TODO: adjust sigs when finished
def evo_autotune(configs, key, pop_size=400, max_generation=10, max_tuned_configs=500, top_k=50, cost_model="random", seed=0, do_bench=None, warmup_configs=200):
    def decorator(fn):
        return EvoAutotuner(fn, fn.arg_names, configs, key, pop_size=pop_size, max_generation=max_generation, max_tuned_configs=max_tuned_configs, top_k=top_k, cost_model=cost_model, seed=seed, do_bench=do_bench, warmup_configs=warmup_configs)
    
    return decorator