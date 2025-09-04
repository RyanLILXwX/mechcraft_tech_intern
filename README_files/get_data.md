# Machine Learning Dataset Builder

A small utility that turns your engineered OHLCV feature files into clean, standardized, and properly split train/valid/test datasets for machine-learning experiments. It is designed to plug in after your feature engineering step (e.g., outputs from `feature_engineering.py`) and before model training.

---

## Parameters

- **Flexible input(input_path)**: accept a directory of feature files or one/multiple file paths (CSV/Parquet).
- **Auto target detection(target_col)**: if `target_col` is not provided, the script will pick `target_up_1` when present; otherwise it chooses the first column whose name starts with `target_`.
- **Feature selection(feature_cols)**: use all **non-target numeric columns*- by default, or pass an explicit list via `feature_cols`.
- **split_mode**:
  - `split_mode="time"` (default): preserves temporal order (train -> valid -> test).
  - `split_mode="random"`: randomized split with `sklearn.model_selection.train_test_split`.
- **train_size**: Proportion of data assigned to the training set.
- **valid_size**: Proportion of data assigned to the validation set. The test set share is inferred as 1 - train_size - valid_size.
- **random_state**: Random seed for reproducibility (used when split_mode="random").
- **shuffle_random_split**: Whether to shuffle before splitting when using random mode.
- **scale**: Whether to apply standard scaling (fit on train, apply to valid/test). `StandardScaler` is fit **only on the training set and applied to valid/test to prevent leakage.
- **verbose**: If True, prints progress and dataset summaries.
- **Simple outputs**: returns `(X_train, y_train, X_valid, y_valid, X_test, y_test, meta)`. 
    - meta:
        - "files": list of source file paths
        - "target_col": chosen target column
        - "feature_cols": chosen feature columns
        - "split_mode": chosen split method
        - "sizes": number of samples in each split
        - "scaled": whether scaling was applied

---

## Installation

`pyarrow` enables fast Parquet IO. If your data is only CSV, it’s still a good idea to keep it installed.

```bash
pip install numpy pandas scikit-learn pyarrow
```

---

## Data Assumptions

* Input files are **tabular** (CSV or Parquet) with columns produced by your feature engineering step.
* Target columns are named like `target_*` (e.g., `target_up_1`).
* Timestamps (e.g., `open_time`, `timestamp`) and non-numeric identifiers (e.g., `symbol`) may be present; they’re **excluded** from features unless you explicitly include them in `feature_cols`.
* Rows containing missing values in the **selected** columns are dropped (typical default; adjust in code if you prefer imputation).

---

## Workflow

This script does **only one job**: load processed feature files, resolve the target/feature sets, split the dataset, and scale features.

1. **Input Resolution**

   * `input_path` can be:

     * a directory containing feature files (CSV/Parquet),
     * a single file path,
     * or a list of file paths.
   * The script **does not** fetch raw data from Binance or generate new features. It expects that step to be already done by `binance_downloader.py` and `feature_engineering.py`.

2. **File Loading**

   * Reads all provided files into a single `pandas.DataFrame`.
   * Only **tabular data** is supported (no JSON, no raw API responses).

3. **Target Column Selection**

   * If `target_col` is explicitly set, that column is used.
   * Otherwise:

     * it first looks for `"target_up_1"`,
     * if not found, picks the **first column starting with `target_`**.
   * If no such column exists, the function raises an error.

4. **Feature Column Selection**

   * If `feature_cols` is provided, only those columns are used.
   * If `None`, it automatically selects all **numeric, non-target columns**.
   * Non-numeric metadata (e.g., `symbol`, `timestamp`) is ignored unless explicitly included.

5. **Cleaning**

   * Drops rows with missing values in the **selected target or feature columns**.
   * No interpolation or imputation is performed.

6. **Splitting**

   * `split_mode="time"` (default):

     * sorts rows by timestamp (if available),
     * takes the first `train_size` fraction as training,
     * next `valid_size` as validation,
     * remainder as test.
   * `split_mode="random"`:

     * applies `sklearn.model_selection.train_test_split` with the given proportions.
   * Important: the script does **not** perform cross-validation, rolling windows, or walk-forward splits — just a single train/valid/test partition.

7. **Scaling**

   * Fits a `StandardScaler` **on the training features only**.
   * Applies the same transformation to validation and test sets.
   * Ensures no look-ahead leakage from future data.

8. **Return Values**
   The main function returns a tuple:

   ```python
   (
       X_train, y_train,   # training set
       X_valid, y_valid,   # validation set
       X_test, y_test,     # test set
       meta                # metadata dict
   )
   ```

---

## Functions

- `load_time_index`
- `gather_input_files`
- `concat_intersection_between_dataframes`
- `select_columns`
- `split_by_time`
- `split_randomly`
- `scale_if_needed`
- `get_ml_datasets`
- `DataConfig`

---

## Notes

* **No target found**: ensure at least one `target_*` column exists, or pass `target_col` explicitly.
* **Empty feature set**: check that your files contain numeric feature columns, or pass `feature_cols`.
* **Unbalanced splits**: verify `train_size + valid_size < 1.0`. The remainder becomes the test set.