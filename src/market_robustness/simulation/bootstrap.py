import numpy as np
import pandas as pd


def block_bootstrap(returns: pd.Series, block_size: int, seed: int) -> pd.Series:
    """Generate one simulated alternative history via block bootstrap."""
    rng = np.random.default_rng(seed=seed)
    N = len(returns)
    blocks = []
    count = 0
    while count < N:
        x = rng.integers(0, N - block_size + 1)
        y = returns.iloc[x : x + block_size]
        blocks.append(y)
        count += block_size

    combined = pd.concat(blocks)
    trimmed = combined.iloc[:N]
    trimmed = trimmed.reset_index(drop=True)
    return trimmed



