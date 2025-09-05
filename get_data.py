from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

@dataclass
class DataConfig:
    # Input can be a directory of feature files or one/multiple files
    input_path: Union[str, List[str]]
    # Name of the target column. If None, auto-pick 'target_up_1' when it present. Otherwise, choose the first 'target_*'
    target_col: Optional[str] = None
    # Optional explicit feature columns. If None, use all non-target numeric columns
    feature_cols: Optional[List[str]] = None

    # Split settings
    # Data splitting method. "time" means splitting by time (training set is in the past, validation/testing set is in the future).
    split_mode: str = "time"
    # The default split is 70% train, 15% valid, 15% test
    train_size: float = 0.7
    valid_size: float = 0.15
    # To ensure that random partitioning is reproducible
    random_state: int = 42
    # Whether to shuffle the data when randomly splitting. Usually should be True.
    shuffle_random_split: bool = True

    # Scaling
    # Standardize features based on train only, then apply to valid and test data. Avoid data leakage.
    scale: bool = True
    
    # Whether we need to print out the progress information.
    verbose: bool = True

def load_time_index(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load a single features file (CSV or Parquet) and attempt to preserve a time index.
    This helper reads a features file from disk and returns a DataFrame suitable for
    time-series work. For both Parquet and CSV files, if a 'ts' column is present, it is
    parsed as UTC datetimes and used as a sorted DateTimeIndex.

   Args:
        path (Union[str, Path]): Path to the features file (CSV or Parquet).

    Returns:
        pandas.DataFrame: The loaded features table. For CSV, the first column becomes
            the (datetime) index. For Parquet, if a 'ts' column exists, it is parsed
            to UTC datetimes and set as a sorted DateTimeIndex; otherwise the default
            index is preserved.
    """
    p = Path(path)
    if (not p.exists()):
        raise FileNotFoundError(f"File not found: {p}")
    if (p.suffix.lower() == ".csv"):
        df = pd.read_csv(p)
        # Drop any "Unnamed: *" columns that pandas sometimes adds
        for c in list(df.columns):
            if (str(c).startswith("Unnamed")):
                df = df.drop(columns=[c])
        # Explicitly convert ts to datetime and set it as index
        if ("ts" in df.columns):
            try:
                df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
                df = df.set_index("ts").sort_index()
            except Exception:
                pass
    elif (p.suffix.lower() in (".parquet", ".pq")):
        df = pd.read_parquet(p)
        # If index looks like a timestamp column in data, try to make a DateTimeIndex
        if ("ts" in df.columns):
            try:
                df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
                df = df.set_index("ts").sort_index()
            except Exception:
                pass
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}")
    return df

def gather_input_files(input_path: Union[str, List[str]]) -> List[Path]:
    """
    Resolve input paths into a standardized list of files.

    Args:
        input_path (Union[str, List[str]]): Either a single file path (string), a directory path containing 
        feature files, or a list of file paths.

    Returns:
        List[Path]: A list of Path objects pointing to the resolved input files, sorted if coming from a directory.
    """
    # ["./data/xxx.csv", "./data/yyy.csv"]
    if (isinstance(input_path, list)):
        return [Path(x) for x in input_path]
    p = Path(input_path)
    # "./data"
    if (p.is_dir()):
        files = sorted(list(p.glob("*.parquet")) + list(p.glob("*.csv")))
        if (not files):
            raise FileNotFoundError(f"No feature files found in: {p}")
        return files
    # "./data/xxx.csv"
    if (p.is_file()):
        return [p]
    raise FileNotFoundError(f"Input not found: {p}")

def concat_intersection_between_dataframes(dataframes: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenate multiple DataFrames using only their shared columns. This function takes a list of pandas DataFrames, identifies the 
    intersection of their column sets, and concatenates them row-wise based on these common columns. The resulting DataFrame preserves 
    the original indices of all inputs and is returned with its rows sorted by index. Columns not present in all inputs are discarded.

    Args:
        dataframes (List[pd.DataFrame]): A list of pandas DataFrames to be concatenated. 
        Each DataFrame may have a different set of columns.

    Returns:
        pd.DataFrame: A single DataFrame formed by vertically stacking all inputs on their shared columns, with indices preserved and sorted.
    """
    if (len(dataframes) == 1):
        return dataframes[0].copy()
    common = set(dataframes[0].columns)
    for d in dataframes[1:]:
        # common = common.intersection(set(d.columns))
        common &= set(d.columns)
    columns = sorted(common)
    out = pd.concat([d[columns] for d in dataframes], axis=0, ignore_index=False)
    return out.sort_index()

def select_columns(df: pd.DataFrame, target_col: Optional[str], feature_cols: Optional[List[str]]) -> Tuple[str, List[str]]:
    """
    Select the target column and feature columns for a supervised task.
    Determines the target (y) and feature set (X) from a DataFrame using
    simple, consistent rules. If `target_col` is not provided, the function
    prefers 'target_up_1' when available; otherwise it picks the first
    column whose name starts with 'target_'. For features, if `feature_cols`
    is not provided, it automatically selects all numeric columns that are
    not target-like (i.e., not starting with 'target_'). If `feature_cols`
    is provided, it validates their existence without enforcing dtypes.

    Args:
        df (pd.DataFrame): Input DataFrame containing target and feature candidates.
        target_col (Optional[str]): Explicit target column name. If None, the function auto-detects 'target_up_1' or the first 'target_*' column.
        feature_cols (Optional[List[str]]): Explicit feature column names. If None, automatically selects all numeric, non-target columns.

    Returns:
        Tuple[str, List[str]]: A tuple of (target_column_name, feature_column_names).
    """
    # Target
    if (target_col is None):
        if ("target_up_1" in df.columns):
            target_col = "target_up_1"
        else:
            cands = [c for c in df.columns if (c.startswith("target_"))]
            if (not cands):
                raise ValueError("No target columns found (prefix 'target_').")
            target_col = cands[0]
    if (target_col not in df.columns):
        raise ValueError(f"target_col '{target_col}' not in dataframe.")
    # Features
    if (feature_cols is None):
        targets = [c for c in df.columns if (c.startswith("target_"))]
        # Check if columns not in targets list and is a nummeric type
        features = [c for c in df.columns if (c not in targets and np.issubdtype(df[c].dtype, np.number))]
    else:
        features = feature_cols
        miss = [c for c in features if c not in df.columns]
        if (miss != []):
            raise ValueError(f"Missing feature columns: {miss}")
    if (not features):
        raise ValueError("No feature columns selected.")
    return target_col, features


def split_by_time(idx: pd.Index, n: int, train_size: float, valid_size: float) -> Tuple[pd.Index, pd.Index, pd.Index]:
    """
    Split an index into train/validation/test partitions in chronological order.
    Performs a time-based split of a pandas Index (e.g., DatetimeIndex or 
    RangeIndex) according to the given train and validation proportions. 
    The split is sequential, so the earliest elements go to the training 
    set, followed by the validation set, and the most recent elements 
    go to the test set.

    Args:
        idx (pd.Index): The index to be partitioned, assumed to be ordered chronologically.
        n (int): The total number of elements in the index (typically len(idx)).
        train_size (float): Proportion of the data to allocate to the training set (0-1).
        valid_size (float): Proportion of the data to allocate to the validation set (0-1). 
        Then the test size is inferred as 1 - train_size - valid_size.

    Returns:
        Tuple[pd.Index, pd.Index, pd.Index]: Three index slices corresponding to (train_index, valid_index, test_index).
    """
    n_train = int(n * train_size)
    n_valid = int(n * (train_size + valid_size))
    return idx[:n_train], idx[n_train: n_valid], idx[n_valid:]


def split_randomly(n: int, train_size: float, valid_size: float, random_state: int, shuffle: bool, index: pd.Index) -> Tuple[pd.Index, pd.Index, pd.Index]:
    """
    Perform a random train/validation/test split of an index.
    Splits the given index into train, validation, and test subsets using
    scikit-learn's `train_test_split`. The split is random, controlled by
    the specified proportions and random seed. First, the test set is
    separated based on its inferred proportion (1 - train_size - valid_size).
    Then the remaining data is split into train and validation sets using
    the relative ratio of `valid_size / (train_size + valid_size)`.

    Args:
        n (int): Total number of samples (typically len(index)).
        train_size (float): Proportion of data to assign to the training set (0-1).
        valid_size (float): Proportion of data to assign to the validation set (0-1).
        random_state (int): Random seed for reproducibility of splits.
        shuffle (bool): Whether to shuffle the data before splitting.
        index (pd.Index): The original pandas Index to be partitioned. Subsets will retain its indexing.

    Returns:
        Tuple[pd.Index, pd.Index, pd.Index]: Three subsets of the original index corresponding to (train_index, valid_index, test_index).
    """
    test_size = 1.0 - train_size - valid_size
    if (test_size <= 0):
        raise ValueError("train_size + valid_size must be < 1.0")
    idx_all = np.arange(n)
    idx_temp, idx_test = train_test_split(idx_all, test_size=test_size, random_state=random_state, shuffle=shuffle)
    valid_ratio = valid_size / (train_size + valid_size)
    idx_train, idx_valid = train_test_split(idx_temp, test_size=valid_ratio, random_state=random_state, shuffle=shuffle)
    return index[idx_train], index[idx_valid], index[idx_test]

def scale_if_needed(X_train: pd.DataFrame, X_valid: pd.DataFrame, X_test: pd.DataFrame, enable: bool):
    """
    Optionally apply standard scaling (z-score normalization) to datasets.
    Fits a scikit-learn StandardScaler on the training features and applies
    the same transformation to the validation and test sets. This ensures
    that scaling parameters (mean and standard deviation) are derived only
    from the training data, avoiding information leakage. If scaling is
    disabled, the original datasets are returned unchanged.

    Args:
        X_train (pd.DataFrame): Training feature matrix.
        X_valid (pd.DataFrame): Validation feature matrix.
        X_test (pd.DataFrame): Test feature matrix.
        enable (bool): If True, perform scaling. if False, return the inputs unchanged.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[StandardScaler]]:
            - Scaled training DataFrame
            - Scaled validation DataFrame
            - Scaled test DataFrame
            - The fitted StandardScaler instance (or None if scaling is disabled)
    """
    if (not enable):
        return X_train, X_valid, X_test, None
    scaler = StandardScaler()
    Xtr = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
    Xva = pd.DataFrame(scaler.transform(X_valid), index=X_valid.index, columns=X_valid.columns)
    Xte = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
    return Xtr, Xva, Xte, scaler

def get_ml_datasets(input_path: Union[str, List[str]], target_col: Optional[str] = None, 
                    feature_cols: Optional[List[str]] = None, split_mode: str = "time", 
                    train_size: float = 0.7, valid_size: float = 0.15, random_state: int = 42, 
                    shuffle_random_split: bool = True, scale: bool = True, verbose: bool = True):
    """
    Load feature data and prepare train/validation/test datasets for machine learning models.
    This function orchestrates the end-to-end preparation of datasets for 
    supervised learning. It loads feature files, aligns them on shared columns, 
    selects the target and feature columns, removes rows with missing values, 
    splits the data into train/validation/test sets (either by time order or 
    randomly), optionally standardizes the features, and returns the resulting 
    datasets along with metadata.

    Args:
        input_path (Union[str, List[str]]):
            Path to feature data, either a directory containing files or a list 
            of file paths.
        target_col (Optional[str], default=None):
            Target column to use. If None, the function will auto-detect 
            'target_up_1' or the first column starting with 'target_'.
        feature_cols (Optional[List[str]], default=None):
            Explicit list of feature columns to use. If None, automatically 
            selects all numeric, non-target columns.
        split_mode (str, default="time"):
            How to split the dataset: "time" for chronological split, or 
            "random" for shuffled split using sklearn.
        train_size (float, default=0.7):
            Proportion of data assigned to the training set.
        valid_size (float, default=0.15):
            Proportion of data assigned to the validation set. The test set 
            share is inferred as 1 - train_size - valid_size.
        random_state (int, default=42):
            Random seed for reproducibility (used when split_mode="random").
        shuffle_random_split (bool, default=True):
            Whether to shuffle before splitting when using random mode.
        scale (bool, default=True):
            Whether to apply standard scaling (fit on train, apply to valid/test).
        verbose (bool, default=True):
            If True, prints progress and dataset summaries.

    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, dict]:
            - X_train: Training feature DataFrame
            - y_train: Training target Series
            - X_valid: Validation feature DataFrame
            - y_valid: Validation target Series
            - X_test: Test feature DataFrame
            - y_test: Test target Series
            - meta: Dictionary of metadata, including:
                - "files": list of source file paths
                - "target_col": chosen target column
                - "feature_cols": chosen feature columns
                - "split_mode": chosen split method
                - "sizes": number of samples in each split
                - "scaled": whether scaling was applied
    """
    # Load files
    files = gather_input_files(input_path)
    if (verbose):
        print(f"[get_data] loading {len(files)} file(s)")
    dfs = [load_time_index(p) for p in files]
    df = concat_intersection_between_dataframes(dfs)
    # Column selection
    target_col, features = select_columns(df, target_col, feature_cols)
    # Drop rows with NA on needed cols
    cols = features + [target_col]
    data = df[cols].dropna()
    if (not data.index.is_monotonic_increasing):
        data = data.sort_index()
    X = data[features]
    y = data[target_col]
    # Split data by time or randomly
    if (split_mode == "time"):
        idx_tr, idx_va, idx_te = split_by_time(X.index, len(X), train_size, valid_size)
    elif (split_mode == "random"):
        idx_tr, idx_va, idx_te = split_randomly(len(X), train_size, valid_size, random_state, shuffle_random_split, X.index)
    else:
        raise ValueError("split_mode must be 'time' or 'random'")
    X_train, y_train = X.loc[idx_tr], y.loc[idx_tr]
    X_valid, y_valid = X.loc[idx_va], y.loc[idx_va]
    X_test,  y_test  = X.loc[idx_te], y.loc[idx_te]
    # Scale if needed
    X_train, X_valid, X_test, scaler = scale_if_needed(X_train, X_valid, X_test, scale)
    # Meta
    # bool(scaler is not None) checked that if scaling actually performed in the function?
    meta = {
        "files": [str(f) for f in files],
        "target_col": target_col,
        "feature_cols": features,
        "split_mode": split_mode,
        "sizes": {"train": len(X_train), "valid": len(X_valid), "test": len(X_test)},
        "scaled": bool(scaler is not None)}
    if (verbose):
        print(f"[get_data] done: train={len(X_train)}, valid={len(X_valid)}, test={len(X_test)} | target={target_col} | features={len(features)}")
    return X_train, y_train, X_valid, y_valid, X_test, y_test, meta