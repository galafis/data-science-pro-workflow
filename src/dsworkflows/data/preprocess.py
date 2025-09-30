"""Data Preprocessing Module.

This module provides functions for data preprocessing, including cleaning,
encoding, and imputation. It uses sklearn's synthetic datasets for demonstration.

Examples:
    >>> from dsworkflows.data.preprocess import clean_data, encode_categorical
    >>> from sklearn.datasets import make_classification
    >>> import pandas as pd
    >>> X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    >>> df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
    >>> cleaned_df = clean_data(df)
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, List
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder


def clean_data(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
    drop_na_threshold: float = 0.5
) -> pd.DataFrame:
    """Clean dataset by removing duplicates and handling missing values.
    
    Args:
        df: Input DataFrame to clean.
        drop_duplicates: Whether to remove duplicate rows.
        drop_na_threshold: Drop columns with more than this fraction of missing values.
    
    Returns:
        Cleaned DataFrame.
    
    Examples:
        >>> from sklearn.datasets import make_classification
        >>> import pandas as pd
        >>> import numpy as np
        >>> X, _ = make_classification(n_samples=100, n_features=5, random_state=42)
        >>> df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
        >>> # Add some missing values
        >>> df.iloc[0:5, 0] = np.nan
        >>> cleaned = clean_data(df, drop_na_threshold=0.1)
        >>> print(f"Original shape: {df.shape}, Cleaned shape: {cleaned.shape}")
    """
    df_clean = df.copy()
    
    # Remove duplicates
    if drop_duplicates:
        df_clean = df_clean.drop_duplicates()
    
    # Drop columns with too many missing values
    missing_fraction = df_clean.isnull().sum() / len(df_clean)
    cols_to_keep = missing_fraction[missing_fraction <= drop_na_threshold].index
    df_clean = df_clean[cols_to_keep]
    
    return df_clean


def impute_missing_values(
    df: pd.DataFrame,
    strategy: str = 'mean',
    fill_value: Optional[Union[str, int, float]] = None
) -> pd.DataFrame:
    """Impute missing values in numerical columns.
    
    Args:
        df: Input DataFrame with missing values.
        strategy: Imputation strategy - 'mean', 'median', 'most_frequent', or 'constant'.
        fill_value: Value to use when strategy is 'constant'.
    
    Returns:
        DataFrame with imputed values.
    
    Examples:
        >>> from sklearn.datasets import make_regression
        >>> import pandas as pd
        >>> import numpy as np
        >>> X, _ = make_regression(n_samples=100, n_features=4, random_state=42)
        >>> df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(4)])
        >>> # Introduce missing values
        >>> df.iloc[0:10, 0] = np.nan
        >>> df.iloc[5:15, 1] = np.nan
        >>> imputed = impute_missing_values(df, strategy='mean')
        >>> print(f"Missing values before: {df.isnull().sum().sum()}")
        >>> print(f"Missing values after: {imputed.isnull().sum().sum()}")
    """
    df_imputed = df.copy()
    
    # Separate numerical and non-numerical columns
    numerical_cols = df_imputed.select_dtypes(include=[np.number]).columns
    
    if len(numerical_cols) > 0:
        imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
        df_imputed[numerical_cols] = imputer.fit_transform(df_imputed[numerical_cols])
    
    return df_imputed


def encode_categorical(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = 'onehot'
) -> pd.DataFrame:
    """Encode categorical variables.
    
    Args:
        df: Input DataFrame with categorical columns.
        columns: List of column names to encode. If None, encode all object columns.
        method: Encoding method - 'onehot' or 'label'.
    
    Returns:
        DataFrame with encoded categorical variables.
    
    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'category_a': ['cat', 'dog', 'cat', 'bird'],
        ...     'category_b': ['red', 'blue', 'red', 'green'],
        ...     'value': [1, 2, 3, 4]
        ... })
        >>> encoded = encode_categorical(df, columns=['category_a', 'category_b'])
        >>> print(f"Original columns: {df.columns.tolist()}")
        >>> print(f"Encoded columns: {encoded.columns.tolist()}")
    """
    df_encoded = df.copy()
    
    # Identify columns to encode
    if columns is None:
        columns = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if method == 'onehot':
        df_encoded = pd.get_dummies(df_encoded, columns=columns, drop_first=True)
    elif method == 'label':
        for col in columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    else:
        raise ValueError(f"Unknown encoding method: {method}")
    
    return df_encoded


def scale_features(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = 'standard'
) -> pd.DataFrame:
    """Scale numerical features.
    
    Args:
        df: Input DataFrame with numerical columns.
        columns: List of column names to scale. If None, scale all numerical columns.
        method: Scaling method - currently only 'standard' (z-score normalization).
    
    Returns:
        DataFrame with scaled features.
    
    Examples:
        >>> from sklearn.datasets import make_regression
        >>> import pandas as pd
        >>> X, _ = make_regression(n_samples=100, n_features=3, random_state=42)
        >>> df = pd.DataFrame(X, columns=['feature_0', 'feature_1', 'feature_2'])
        >>> scaled = scale_features(df)
        >>> print(f"Original mean: {df.mean().values}")
        >>> print(f"Scaled mean: {scaled.mean().values}")
        >>> print(f"Scaled std: {scaled.std().values}")
    """
    df_scaled = df.copy()
    
    # Identify columns to scale
    if columns is None:
        columns = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
    
    if method == 'standard':
        scaler = StandardScaler()
        df_scaled[columns] = scaler.fit_transform(df_scaled[columns])
    else:
        raise ValueError(f"Unknown scaling method: {method}")
    
    return df_scaled


def remove_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = 'iqr',
    threshold: float = 1.5
) -> pd.DataFrame:
    """Remove outliers from numerical columns.
    
    Args:
        df: Input DataFrame.
        columns: Columns to check for outliers. If None, check all numerical columns.
        method: Outlier detection method - 'iqr' (Interquartile Range) or 'zscore'.
        threshold: Threshold for outlier detection (1.5 for IQR, 3.0 for z-score).
    
    Returns:
        DataFrame with outliers removed.
    
    Examples:
        >>> from sklearn.datasets import make_regression
        >>> import pandas as pd
        >>> import numpy as np
        >>> X, _ = make_regression(n_samples=100, n_features=3, random_state=42)
        >>> df = pd.DataFrame(X, columns=['feature_0', 'feature_1', 'feature_2'])
        >>> # Add some outliers
        >>> df.iloc[0, 0] = 1000
        >>> df.iloc[1, 1] = -1000
        >>> cleaned = remove_outliers(df, method='iqr', threshold=1.5)
        >>> print(f"Original shape: {df.shape}, After outlier removal: {cleaned.shape}")
    """
    df_clean = df.copy()
    
    # Identify columns to check
    if columns is None:
        columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    
    if method == 'iqr':
        for col in columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df_clean = df_clean[
                (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
            ]
    elif method == 'zscore':
        for col in columns:
            z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
            df_clean = df_clean[z_scores < threshold]
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")
    
    return df_clean


if __name__ == "__main__":
    # Demo with synthetic data
    from sklearn.datasets import make_classification
    
    print("=== Data Preprocessing Demo ===")
    print("\n1. Creating synthetic dataset...")
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        random_state=42
    )
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
    df['target'] = y
    
    # Add some issues
    df.iloc[0:5, 0] = np.nan
    df.iloc[10:12, :] = df.iloc[10:12, :].values  # Duplicates
    df.iloc[0, 1] = 1000  # Outlier
    
    print(f"Original shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    print("\n2. Cleaning data...")
    df_clean = clean_data(df, drop_na_threshold=0.9)
    print(f"After cleaning: {df_clean.shape}")
    
    print("\n3. Imputing missing values...")
    df_imputed = impute_missing_values(df_clean, strategy='mean')
    print(f"Missing values after imputation: {df_imputed.isnull().sum().sum()}")
    
    print("\n4. Removing outliers...")
    df_no_outliers = remove_outliers(df_imputed, method='iqr', threshold=1.5)
    print(f"Shape after outlier removal: {df_no_outliers.shape}")
    
    print("\n5. Scaling features...")
    feature_cols = [col for col in df_no_outliers.columns if col != 'target']
    df_scaled = scale_features(df_no_outliers, columns=feature_cols)
    print(f"Scaled features mean: {df_scaled[feature_cols].mean().values}")
    print(f"Scaled features std: {df_scaled[feature_cols].std().values}")
    
    print("\n=== Demo completed ===")
