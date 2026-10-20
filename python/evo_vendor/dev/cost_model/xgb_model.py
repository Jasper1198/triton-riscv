from __future__ import annotations

from typing import Callable, Optional, List, Any, Tuple
import numpy as np
import xgboost as xgb
from triton.runtime.autotuner import Config

from .cost_model import CostModel


class XGBCostModel(CostModel):
    def __init__(self,
        feature_order: List[str],
        seed: int = 0,
        n_estimators : int = 100,    #10000
        max_depth: int = 10,         #6
        learning_rate: float = 0.2, #0.15
        gamma: float = 0.001
    ):
        super().__init__()

        # global feature order
        self.feature_order = list(feature_order)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.early_stopping_rounds = 50

        self.params = {
            "objective": "reg:squarederror",
            "max_depth": max_depth,
            "eta" : learning_rate,
            "seed": seed,
            "gamma": gamma,
            "verbosity": 0,
        }

        self.num_boost_round = n_estimators
        self.model = None

        self.is_fitted = False
        self.X_train: List[List[float]] = []
        self.y_train: List[float] = []

    def _dict_to_feature(self, d: dict, extra_features: Optional[dict] = None) -> List[float]:
        merged = dict(d)

        if extra_features is not None:
            merged.update(extra_features)


        feats = []
        grid_name = ['grid_0', 'grid_1', 'grid_2']
        num_programs = 1
        for feat_name in self.feature_order:
            v = merged.get(feat_name, 0)
            if feat_name in grid_name:
                num_programs *= v
            feats.append(float(v))
        # the number of kernel launch will be added to feature
        feats.append(float(num_programs))
        return feats

    def _chromosome_to_feature(self, chrom, extra_features: Optional[dict] = None) -> List[float]:
        """
        Convert one Chromosome into a fixed-order numeric feature vector.
        We sort by gene name so the order is deterministic.
        """
        gene_map = {g.name: g.val for g in chrom.genes}
        return self._dict_to_feature(gene_map, extra_features=extra_features)
    
    def _config_to_feature(self, cfg: Config, extra_features: Optional[dict] = None) -> List[float]:
        return self._dict_to_feature(cfg.all_kwargs(), extra_features=extra_features)

    def _population_to_features(self, population: List, extra_features: Optional[dict] = None) -> np.ndarray:
        feats = [self._chromosome_to_feature(chrom, extra_features=extra_features) for chrom in population]
        return np.asarray(feats, dtype=np.float32)
    
    def _configs_to_feature(self, cfgs: List[Config], extra_features: Optional[dict] = None) -> np.ndarray:
        feats = [self._config_to_feature(cfg, extra_features=extra_features) for cfg in cfgs]
        return np.asarray(feats, dtype=np.float32)

    def score(self, population: Optional[List] = None, extra_features: Optional[List[dict]] = None) -> np.ndarray:
        assert population is not None, "lost population"

        n = len(population)
        if n == 0:
            return np.array([], dtype=np.float64)
        
        if extra_features is not None:
            assert len(extra_features) == n, "extra_features length mismatch"

        if not self.is_fitted:
            raw = self.rng.random(n)
            return raw / raw.sum()
        
        feats = []
        for i, chrom in enumerate(population):
            ef = extra_features[i] if extra_features is not None else None
            feats.append(self._chromosome_to_feature(chrom=chrom, extra_features=ef))
        
        X = np.asarray(feats, dtype=np.float32)
        dtest = xgb.DMatrix(X)

        # predict latency (smaller is better)
        pred_latency = self.model.predict(dtest)

        # convert latency -> fitness (bigger is better)
        fitness = -pred_latency.astype(np.float64)

        # stabilize + normalize to prob
        fitness = fitness - fitness.min() + 1e-6
        s = fitness.sum()
        if s <= 0:
            return np.ones(n, dtype=np.float64) / n
        
        return fitness / s


    def update(self, configs: List[Config], latencies: List[float], extra_features: Optional[List[dict]] = None):
        assert len(configs) == len(latencies), "configs and latencies length mismatch"

        if extra_features is not None:
            assert len(extra_features) == len(configs), "extra_features length mismatch"

        for i, (cfg, lat) in enumerate(zip(configs, latencies)):
            if np.isinf(lat) or np.isnan(lat):
                continue
            ef = extra_features[i] if extra_features is not None else None
            feats = self._config_to_feature(cfg, extra_features=ef)
            self.X_train.append(feats)
            self.y_train.append(float(lat))

        if len(self.y_train) == 0:
            return
        
        X = np.asarray(self.X_train, dtype=np.float32)
        y = np.asarray(self.y_train, dtype=np.float32)

        dtrain = xgb.DMatrix(X, label=y)

        self.model = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=self.num_boost_round,
        )
        self.is_fitted = True

        return