from pathlib import Path
import sys
from binance_downloader import get_binance_dataset, parse_arguments
from feature_engineering import get_features_dataset, WindowsConfig

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

if __name__ == "__main__":
    run_pipeline()