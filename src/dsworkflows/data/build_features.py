"""
Feature engineering utilities for building new features.

Functions:
- polynomial_features: generate polynomial and interaction terms using scikit-learn PolynomialFeatures
- scale_features: scale features using StandardScaler or MinMaxScaler
- cross_features: create crossed features (hashing or cartesian combinations) from categorical columns

Each function includes a synthetic example in the docstring.
"""
from __future__ import annotations

from typing import Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler

ScalerType = Literal["standard", "minmax"]


def polynomial_features(
    X: Union[pd.DataFrame, np.ndarray],
    degree: int = 2,
    include_bias: bool = False,
    interaction_only: bool = False,
    feature_names: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, PolynomialFeatures]:
    """
    Generate polynomial and interaction features from numeric data.

    Parameters
    - X: DataFrame or 2D ndarray with numeric columns
    - degree: max polynomial degree (default=2)
    - include_bias: whether to include the bias column of ones
    - interaction_only: if True, only interaction features are produced (no powers)
    - feature_names: override input feature names (used for output column naming)

    Returns
    - df_poly: pandas DataFrame with transformed features
    - transformer: fitted sklearn.preprocessing.PolynomialFeatures instance

    Example
    >>> import pandas as pd
    >>> from dsworkflows.data.build_features import polynomial_features
    >>> X = pd.DataFrame({"x1": [1.0, 2.0, 3.0], "x2": [0.5, 1.0, 1.5]})
    >>> X_poly, pf = polynomial_features(X, degree=2, include_bias=False)
    >>> list(X_poly.columns)
    ['x1', 'x2', 'x1^2', 'x1 x2', 'x2^2']
    >>> X_poly.shape
    (3, 5)
    """
    if isinstance(X, pd.DataFrame):
        X_values = X.values
        in_names = list(X.columns)
    else:
        X_values = np.asarray(X)
        if X_values.ndim != 2:
            raise ValueError("X must be 2D array-like")
        if feature_names is not None:
            in_names = list(feature_names)
        else:
            in_names = [f"x{i}" for i in range(X_values.shape[1])]

    pf = PolynomialFeatures(
        degree=degree,
        include_bias=include_bias,
        interaction_only=interaction_only,
        order="C",
    )
    Z = pf.fit_transform(X_values)

    # Build readable output names
    try:
        out_names = pf.get_feature_names_out(in_names)
    except Exception:
        out_names = [f"f{i}" for i in range(Z.shape[1])]

    df_poly = pd.DataFrame(Z, columns=out_names, index=(X.index if isinstance(X, pd.DataFrame) else None))
    return df_poly, pf


def scale_features(
    X: Union[pd.DataFrame, np.ndarray],
    scaler: ScalerType = "standard",
    feature_names: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, Union[StandardScaler, MinMaxScaler]]:
    """
    Scale numeric features using StandardScaler or MinMaxScaler.

    Parameters
    - X: DataFrame or 2D ndarray with numeric columns
    - scaler: 'standard' for StandardScaler, 'minmax' for MinMaxScaler
    - feature_names: optional names when X is ndarray

    Returns
    - df_scaled: pandas DataFrame with scaled values
    - transformer: fitted scaler instance

    Example
    >>> import numpy as np
    >>> from dsworkflows.data.build_features import scale_features
    >>> X = np.array([[1., 10.], [2., 20.], [3., 30.]])
    >>> df_std, sc_std = scale_features(X, scaler="standard")
    >>> df_minmax, sc_mm = scale_features(X, scaler="minmax")
    >>> df_std.round(3).values.tolist()
    [[-1.225, -1.225], [0.0, 0.0], [1.225, 1.225]]
    >>> df_minmax.values.tolist()
    [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]
    """
    if isinstance(X, pd.DataFrame):
        X_values = X.values
        names = list(X.columns)
        index = X.index
    else:
        X_values = np.asarray(X)
        if X_values.ndim != 2:
            raise ValueError("X must be 2D array-like")
        names = list(feature_names) if feature_names is not None else [f"x{i}" for i in range(X_values.shape[1])]
        index = None

    if scaler == "standard":
        trf = StandardScaler()
    elif scaler == "minmax":
        trf = MinMaxScaler()
    else:
        raise ValueError("scaler must be 'standard' or 'minmax'")

    X_scaled = trf.fit_transform(X_values)
    df_scaled = pd.DataFrame(X_scaled, columns=names, index=index)
    return df_scaled, trf


def cross_features(
    X: pd.DataFrame,
    columns: Sequence[str],
    method: Literal["cartesian", "hash"] = "cartesian",
    hash_mod: Optional[int] = 1_000_003,
    sep: str = "|",
    new_col_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Create crossed features from categorical columns.

    Methods
    - 'cartesian': join the string representations to form explicit cross tokens
    - 'hash': compute a stable hashed integer of the cross (useful for high-cardinality)

    Parameters
    - X: input DataFrame
    - columns: columns to cross (must be present in X)
    - method: 'cartesian' or 'hash'
    - hash_mod: modulus applied to hashed integers (only for method='hash')
    - sep: separator for cartesian concatenation
    - new_col_name: optional name for the new column; default is 'x_col1_x_col2_...'

    Returns
    - DataFrame with an additional crossed feature column

    Example
    >>> import pandas as pd
    >>> from dsworkflows.data.build_features import cross_features
    >>> df = pd.DataFrame({"city": ["NY", "SF"], "device": ["iOS", "Android"]})
    >>> out = cross_features(df, ["city", "device"], method="cartesian")
    >>> list(out.columns)
    ['city', 'device', 'x_city_x_device']
    >>> out['x_city_x_device'].tolist()
    ['NY|iOS', 'SF|Android']

    Hashing example
    >>> out_h = cross_features(df, ["city", "device"], method="hash", hash_mod=1000)
    >>> out_h['x_city_x_device'].between(0, 999).all()
    True
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame for cross_features")

    for c in columns:
        if c not in X.columns:
            raise KeyError(f"Column '{c}' not in DataFrame")

    if new_col_name is None:
        new_col_name = "x_" + "_x_".join(columns)

    if method == "cartesian":
        crossed = X[columns].astype(str).agg(sep.join, axis=1)
    elif method == "hash":
        if hash_mod is None or hash_mod <= 0:
            raise ValueError("hash_mod must be a positive integer when using method='hash'")
        # Create a tuple string, then hash to a stable integer using pandas hashing utilities
        from pandas.util import hash_pandas_object

        crossed_tokens = X[columns].astype(str)
        hashed = hash_pandas_object(crossed_tokens.apply(tuple, axis=1), index=False).astype(np.int64)
        crossed = (hashed % int(hash_mod)).astype(np.int64)
    else:
        raise ValueError("method must be 'cartesian' or 'hash'")

    X_out = X.copy()
    X_out[new_col_name] = crossed
    return X_out


__all__ = [
    "polynomial_features",
    "scale_features",
    "cross_features",
]
