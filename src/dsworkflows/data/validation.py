"""Data Validation Module.

This module provides utilities for validating data schemas and checking for null values
in datasets. It includes functions for schema validation against expected data types
and comprehensive null value detection across various data structures.

Author: Data Science Pro Workflow
Date: September 2025
"""

from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np


def validate_schema(
    data: pd.DataFrame,
    expected_schema: Dict[str, type],
    strict: bool = True
) -> Dict[str, Any]:
    """Validate DataFrame schema against expected data types.
    
    This function checks if the columns in a DataFrame match the expected schema,
    validating both column names and data types. It can operate in strict mode
    (all columns must match) or lenient mode (only check specified columns).
    
    Args:
        data (pd.DataFrame): The DataFrame to validate.
        expected_schema (Dict[str, type]): Dictionary mapping column names to expected
            data types (e.g., {'age': int, 'name': str, 'salary': float}).
        strict (bool, optional): If True, requires exact match of all columns.
            If False, only validates columns present in expected_schema. Defaults to True.
    
    Returns:
        Dict[str, Any]: A dictionary containing validation results with keys:
            - 'valid' (bool): True if validation passed, False otherwise.
            - 'errors' (List[str]): List of error messages if validation failed.
            - 'missing_columns' (List[str]): Columns in schema but not in data.
            - 'extra_columns' (List[str]): Columns in data but not in schema (strict mode only).
            - 'type_mismatches' (Dict[str, tuple]): Columns with type mismatches.
    
    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'age': [25, 30, 35],
        ...     'name': ['Alice', 'Bob', 'Charlie'],
        ...     'salary': [50000.0, 60000.0, 70000.0]
        ... })
        >>> schema = {'age': int, 'name': str, 'salary': float}
        >>> result = validate_schema(df, schema)
        >>> print(result['valid'])
        True
        >>> 
        >>> # Example with type mismatch
        >>> df_invalid = pd.DataFrame({'age': ['25', '30', '35']})  # age as string
        >>> schema = {'age': int}
        >>> result = validate_schema(df_invalid, schema, strict=False)
        >>> print(result['valid'])
        False
    """
    result = {
        'valid': True,
        'errors': [],
        'missing_columns': [],
        'extra_columns': [],
        'type_mismatches': {}
    }
    
    # Check for missing columns
    data_columns = set(data.columns)
    expected_columns = set(expected_schema.keys())
    missing = expected_columns - data_columns
    
    if missing:
        result['valid'] = False
        result['missing_columns'] = list(missing)
        result['errors'].append(f"Missing columns: {list(missing)}")
    
    # Check for extra columns (strict mode only)
    if strict:
        extra = data_columns - expected_columns
        if extra:
            result['valid'] = False
            result['extra_columns'] = list(extra)
            result['errors'].append(f"Extra columns not in schema: {list(extra)}")
    
    # Check data types for existing columns
    type_mapping = {
        int: [np.int8, np.int16, np.int32, np.int64, int],
        float: [np.float16, np.float32, np.float64, float],
        str: [np.object_, object, str],
        bool: [np.bool_, bool]
    }
    
    for col, expected_type in expected_schema.items():
        if col in data.columns:
            actual_type = data[col].dtype.type
            valid_types = type_mapping.get(expected_type, [expected_type])
            
            if not any(np.issubdtype(data[col].dtype, t) for t in valid_types):
                result['valid'] = False
                result['type_mismatches'][col] = (str(data[col].dtype), expected_type.__name__)
                result['errors'].append(
                    f"Column '{col}' has type {data[col].dtype}, expected {expected_type.__name__}"
                )
    
    return result


def check_nulls(
    data: Union[pd.DataFrame, pd.Series],
    columns: Optional[List[str]] = None,
    threshold: float = 0.0
) -> Dict[str, Any]:
    """Check for null values in a DataFrame or Series.
    
    This function performs comprehensive null value detection, including NaN, None,
    and other missing value representations. It provides detailed statistics about
    null values and can flag columns exceeding a specified threshold.
    
    Args:
        data (Union[pd.DataFrame, pd.Series]): The data to check for null values.
        columns (Optional[List[str]], optional): Specific columns to check. If None,
            checks all columns. Defaults to None.
        threshold (float, optional): Maximum acceptable proportion of null values (0-1).
            Columns exceeding this threshold will be flagged. Defaults to 0.0.
    
    Returns:
        Dict[str, Any]: A dictionary containing null value analysis with keys:
            - 'has_nulls' (bool): True if any null values found.
            - 'total_nulls' (int): Total number of null values.
            - 'null_counts' (Dict[str, int]): Null count per column.
            - 'null_percentages' (Dict[str, float]): Null percentage per column.
            - 'flagged_columns' (List[str]): Columns exceeding threshold.
            - 'clean_columns' (List[str]): Columns with no null values.
    
    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> df = pd.DataFrame({
        ...     'age': [25, np.nan, 35, 40],
        ...     'name': ['Alice', 'Bob', None, 'Diana'],
        ...     'salary': [50000, 60000, 70000, 80000]
        ... })
        >>> result = check_nulls(df)
        >>> print(f"Has nulls: {result['has_nulls']}")
        Has nulls: True
        >>> print(f"Total nulls: {result['total_nulls']}")
        Total nulls: 2
        >>> print(f"Null counts: {result['null_counts']}")
        Null counts: {'age': 1, 'name': 1, 'salary': 0}
        >>> 
        >>> # Check specific columns with threshold
        >>> result = check_nulls(df, columns=['age', 'name'], threshold=0.2)
        >>> print(f"Flagged columns: {result['flagged_columns']}")
        Flagged columns: ['age', 'name']
    """
    # Convert Series to DataFrame for uniform processing
    if isinstance(data, pd.Series):
        data = data.to_frame()
    
    # Determine which columns to check
    cols_to_check = columns if columns is not None else data.columns.tolist()
    
    # Filter to only existing columns
    cols_to_check = [col for col in cols_to_check if col in data.columns]
    
    # Calculate null statistics
    null_counts = {}
    null_percentages = {}
    flagged_columns = []
    clean_columns = []
    
    for col in cols_to_check:
        null_count = data[col].isnull().sum()
        null_counts[col] = int(null_count)
        
        if len(data) > 0:
            null_pct = null_count / len(data)
            null_percentages[col] = round(float(null_pct), 4)
            
            if null_count == 0:
                clean_columns.append(col)
            elif null_pct > threshold:
                flagged_columns.append(col)
        else:
            null_percentages[col] = 0.0
    
    total_nulls = sum(null_counts.values())
    
    result = {
        'has_nulls': total_nulls > 0,
        'total_nulls': total_nulls,
        'null_counts': null_counts,
        'null_percentages': null_percentages,
        'flagged_columns': flagged_columns,
        'clean_columns': clean_columns
    }
    
    return result


def validate_data_quality(
    data: pd.DataFrame,
    schema: Optional[Dict[str, type]] = None,
    null_threshold: float = 0.1,
    strict_schema: bool = False
) -> Dict[str, Any]:
    """Perform comprehensive data quality validation.
    
    This function combines schema validation and null checking to provide a complete
    data quality assessment. It's useful for validating datasets before processing
    or model training.
    
    Args:
        data (pd.DataFrame): The DataFrame to validate.
        schema (Optional[Dict[str, type]], optional): Expected schema for validation.
            If None, skips schema validation. Defaults to None.
        null_threshold (float, optional): Maximum acceptable null proportion.
            Defaults to 0.1 (10%).
        strict_schema (bool, optional): If True, enforces exact schema match.
            Defaults to False.
    
    Returns:
        Dict[str, Any]: A dictionary containing complete validation results with keys:
            - 'valid' (bool): True if all validations passed.
            - 'schema_validation' (Dict): Results from schema validation (if schema provided).
            - 'null_analysis' (Dict): Results from null value checking.
            - 'summary' (str): Human-readable summary of validation results.
    
    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> df = pd.DataFrame({
        ...     'age': [25, 30, 35],
        ...     'name': ['Alice', 'Bob', 'Charlie'],
        ...     'salary': [50000.0, 60000.0, 70000.0]
        ... })
        >>> schema = {'age': int, 'name': str, 'salary': float}
        >>> result = validate_data_quality(df, schema=schema, null_threshold=0.0)
        >>> print(result['valid'])
        True
        >>> print(result['summary'])
        Data quality validation passed. No issues detected.
    """
    result = {
        'valid': True,
        'schema_validation': None,
        'null_analysis': None,
        'summary': ''
    }
    
    issues = []
    
    # Perform schema validation if schema provided
    if schema is not None:
        schema_result = validate_schema(data, schema, strict=strict_schema)
        result['schema_validation'] = schema_result
        
        if not schema_result['valid']:
            result['valid'] = False
            issues.append(f"Schema validation failed: {', '.join(schema_result['errors'])}")
    
    # Perform null checking
    null_result = check_nulls(data, threshold=null_threshold)
    result['null_analysis'] = null_result
    
    if null_result['flagged_columns']:
        result['valid'] = False
        issues.append(
            f"Columns exceed null threshold ({null_threshold*100}%): {null_result['flagged_columns']}"
        )
    
    # Generate summary
    if result['valid']:
        result['summary'] = "Data quality validation passed. No issues detected."
    else:
        result['summary'] = "Data quality validation failed. Issues: " + "; ".join(issues)
    
    return result
