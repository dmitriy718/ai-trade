#!/usr/bin/env python3
"""
Parameter Tuner - Backtest-driven grid search for strategy parameters.

Usage:
  python param_tune.py --csv data/btc.csv --pair BTC/USD --strategy keltner --write-config
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import numpy as np

from src.core.logger import get_logger
from src.ml.backtester import Backtester
from src.strategies.breakout import BreakoutStrategy
from src.strategies.keltner import KeltnerStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.momentum import MomentumStrategy
from src.strategies.reversal import ReversalStrategy
from src.strategies.trend import TrendStrategy

logger = get_logger("param_tuner")


STRATEGY_MAP = {
    "keltner": KeltnerStrategy,
    "trend": TrendStrategy,
    "mean_reversion": MeanReversionStrategy,
    "momentum": MomentumStrategy,
    "breakout": BreakoutStrategy,
    "reversal": ReversalStrategy,
}

DEFAULT_GRIDS: Dict[str, Dict[str, List[Any]]] = {
    "keltner": {
        "ema_period": [20, 30],
        "atr_period": [14],
        "kc_multiplier": [1.5, 2.0],
        "rsi_long_max": [40, 45],
        "rsi_short_min": [60, 55],
    },
    "trend": {
        "ema_fast": [10, 20],
        "ema_slow": [30, 50],
        "adx_threshold": [25, 30],
    },
    "mean_reversion": {
        "bb_period": [20, 50],
        "bb_std": [2.0, 2.5],
        "rsi_oversold": [25, 30],
        "rsi_overbought": [70, 75],
    },
    "momentum": {
        "rsi_threshold": [50, 55],
        "volume_multiplier": [1.5, 2.0],
    },
    "breakout": {
        "lookback_period": [20, 60],
        "volume_confirmation": [1.3, 1.8],
    },
    "reversal": {
        "rsi_extreme_low": [15, 20],
        "rsi_extreme_high": [80, 85],
        "confirmation_candles": [3, 5],
    },
}


def _grid_to_params(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    combos = itertools.product(*values)
    return [dict(zip(keys, combo)) for combo in combos]


def _score_result(result) -> float:
    """Score function: favor Sharpe + return, penalize drawdown."""
    return (result.sharpe_ratio or 0) + (result.total_return_pct / 100.0) - (result.max_drawdown * 2)


async def run_grid(
    pair: str,
    df: pd.DataFrame,
    strategy_name: str,
    grid: Dict[str, List[Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    strategy_cls = STRATEGY_MAP[strategy_name]
    params_list = _grid_to_params(grid)

    backtester = Backtester()
    best_score = float("-inf")
    best_params = {}
    best_metrics = {}
    results: List[Dict[str, Any]] = []

    for params in params_list:
        strategy = strategy_cls(**params)
        result = await backtester.run(
            pair=pair,
            ohlcv_data=df,
            strategies=[strategy],
            confluence_threshold=1,
        )
        score = _score_result(result)
        metrics = result.to_dict()
        results.append({
            "params": params,
            "score": score,
            "metrics": metrics,
        })
        if score > best_score:
            best_score = score
            best_params = params
            best_metrics = metrics

    results.sort(key=lambda r: r["score"], reverse=True)
    return best_params, best_metrics, results


def _write_config(config_path: Path, strategy_name: str, params: Dict[str, Any]) -> Path:
    try:
        from ruamel.yaml import YAML
        yaml = YAML()
        yaml.preserve_quotes = True
        data = yaml.load(config_path.read_text()) if config_path.exists() else {}
        if data is None:
            data = {}
        data.setdefault("strategies", {})
        data["strategies"].setdefault(strategy_name, {})
        data["strategies"][strategy_name].update(params)
        with config_path.open("w") as f:
            yaml.dump(data, f)
        return config_path
    except Exception:
        import yaml
        overlay_path = config_path.with_suffix(".overrides.yaml")
        data = yaml.safe_load(overlay_path.read_text()) if overlay_path.exists() else {}
        data.setdefault("strategies", {})
        data["strategies"].setdefault(strategy_name, {})
        data["strategies"][strategy_name].update(params)
        overlay_path.write_text(yaml.safe_dump(data, sort_keys=False))
        return overlay_path


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="CSV path with OHLCV data")
    parser.add_argument("--pair", required=True, help="Trading pair label (e.g., BTC/USD)")
    parser.add_argument("--strategy", required=True, choices=STRATEGY_MAP.keys())
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--write-config", action="store_true")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--report", default="", help="Path for JSON report (default: tuning_reports/...)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_csv(args.csv)
    # Normalize column names
    df.columns = [c.lower() for c in df.columns]
    required = {"time", "open", "high", "low", "close", "volume"}
    if not required.issubset(set(df.columns)):
        raise SystemExit(f"CSV missing required columns: {sorted(required)}")

    if args.max_rows and len(df) > args.max_rows:
        df = df.tail(args.max_rows)

    grid = DEFAULT_GRIDS.get(args.strategy)
    if not grid:
        raise SystemExit("No grid defined for strategy")

    best_params, best_metrics, results = await run_grid(args.pair, df, args.strategy, grid)

    print("\nBest Params:")
    print(json.dumps(best_params, indent=2))
    print("\nBest Metrics:")
    print(json.dumps(best_metrics, indent=2))

    report_path = Path(args.report) if args.report else Path("tuning_reports") / f"{args.strategy}_{int(time.time())}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "strategy": args.strategy,
        "pair": args.pair,
        "seed": args.seed,
        "grid": grid,
        "best_params": best_params,
        "best_metrics": best_metrics,
        "top_results": results[:max(1, args.top)],
        "total_evaluated": len(results),
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nSaved tuning report to {report_path}.")

    if args.write_config:
        written_path = _write_config(Path(args.config), args.strategy, best_params)
        print(f"\nUpdated {written_path} with best params for {args.strategy}.")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
