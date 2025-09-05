from pathlib import Path
import sys
from binance_downloader import get_binance_dataset, parse_arguments
from feature_engineering import get_features_dataset, WindowsConfig
from get_data import get_ml_datasets

def check_valid(x):
    try:
        return len(x)
    except Exception:
        return "N/A"

def run_pipeline():
    # Parse CLI args
    cfg = parse_arguments()
    # Run downloader
    raw_data = get_binance_dataset()

    # Setup folder layout
    raw_dir = Path(cfg.out_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    features_dir = Path(cfg.out_dir).parent / "processed_data"
    features_dir.mkdir(parents=True, exist_ok=True)
    # Figure out raw file path
    if (isinstance(raw_data, (str, Path))):
        paths = [Path(raw_data)]
    elif (isinstance(raw_data, (list, tuple))):
        paths = [Path(p) for p in raw_data if p]
    else:
        candidates = sorted(raw_dir.glob("*.csv")) + sorted(raw_dir.glob("*.parquet"))
        if (not candidates):
            print("[ERROR] No raw files found.", file=sys.stderr)
            sys.exit(1)
        # Handle all the raw data files
        paths = candidates
    # Build features for each file
    for p in paths:
        print(f"[INFO] Building features for {p.name}")
        df_out, feature_cols, target_cols, saved_path = get_features_dataset(
            csv_path=str(p), 
            interval=cfg.interval, 
            expected_symbol=None, 
            windows_cfg=WindowsConfig(), 
            horizons=(1, 5, 15), 
            task="binclass", 
            thresh=0.0, 
            out_path=str(features_dir), 
            file_format="csv", 
            log_gaps=True)
        print(f"[OK] rows={len(df_out)}, feats={len(feature_cols)}, targets={len(target_cols)}")
        if (saved_path):
            print(f"[Saved] {saved_path}\n")
    
    # We point input_path to the folder containing all processed feature files.
    # Leave target_col=None to auto-pick 'target_up_1' (if available) per your get_data.py logic.
    print("------ [INFO] Building ML datasets from processed features. ------\n")
    ml_result = get_ml_datasets(
        input_path=str(features_dir), 
        target_col=None, # auto-pick, e.g., 'target_up_1' when present
        feature_cols=None, # use all non-target numeric features
        split_mode="time", # time-based split (train past -> valid/test future)
        train_size=0.7, 
        valid_size=0.15, 
        random_state=42, 
        shuffle_random_split=True, # only relevant if split_mode="random"
        scale=True, 
        verbose=True)
    X_train = y_train = X_valid = y_valid = X_test = y_test = meta = None
    if (isinstance(ml_result, tuple)):
        try:
            (X_train, y_train, X_valid, y_valid, X_test, y_test, meta) = ml_result
        except Exception as e:
            print(f"[ERROR] Unexpected ml_result format: {e}", file=sys.stderr)
    # Check the machine learning data and print summary
    print("\n------ Machine Learning Data Summary: ------")
    print(f"Train: X={check_valid(X_train)}, y={check_valid(y_train)}")
    print(f"Valid: X={check_valid(X_valid)}, y={check_valid(y_valid)}")
    print(f"Test : X={check_valid(X_test)},  y={check_valid(y_test)}\n")
    if (meta != None):
        if (isinstance(meta, dict)):
            print("------ Meta information: ------\n")
            print(f"target_col = {meta.get("target_col")}\n")
            print("feature_cols = {}\n".format(meta.get("feature_cols")))
            print("split_mode = {}\n".format(meta.get("split_mode")))
            print("sizes = {}\n".format(meta.get("sizes")))
            print("scaled = {}\n".format(meta.get("scaled")))
        else:
            print("Warning: meta is not a dict object.", file=sys.stderr)
    else:
        print("Warning: meta is None.", file=sys.stderr)
    
    # Call machine learning model

if __name__ == "__main__":
    run_pipeline()