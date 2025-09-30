"""
Data ingestion utilities.
Author: Gabriel Demetrios Lafis (@galafis)
"""
from pathlib import Path
from typing import Optional
import pandas as pd


def read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    path = Path(path)
    return pd.read_csv(path, **kwargs)


def save_parquet(df: pd.DataFrame, path: str | Path, index: bool = False, **kwargs) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=index, **kwargs)


def ingest_example(output_dir: str | Path, n_rows: int = 1000, random_state: int = 42) -> Path:
    """Generate a synthetic binary classification dataset and persist to parquet."""
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=n_rows,
        n_features=12,
        n_informative=6,
        n_redundant=2,
        weights=[0.9, 0.1],
        random_state=random_state,
    )
    cols = [f"f{i:02d}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    df["target"] = y
    out_path = Path(output_dir) / "data" / "raw" / "synthetic_classification.parquet"
    save_parquet(df, out_path)
    return out_path
