import pandas as pd
from pathlib import Path
import yaml

# This defines the project root relative to this file's location
ROOT = Path(__file__).resolve().parents[1]

def load_config():
    """Loads the main config.yaml file."""
    config_path = ROOT / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_raw(name: str):
    """Loads a raw CSV from data/raw."""
    path = ROOT / "data" / "raw" / name
    return pd.read_csv(path)

def save_processed(df, name: str):
    """Saves a processed DataFrame to data/processed."""
    path = ROOT / "data" / "processed" / name
    df.to_csv(path, index=False)
