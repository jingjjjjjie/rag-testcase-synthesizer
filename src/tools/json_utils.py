"""Simple utilities for saving and loading JSON."""

import json
import os

def save_json(data, path, verbose=False):
    """Save data to JSON file."""
    dir_path = os.path.dirname(path)
    if dir_path:  # Only create directory if path includes one
        os.makedirs(dir_path, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    if verbose:
        print(f"Saved {len(data)} items to {path}")


def load_json(path,verbose=False):
    """Load data from JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if verbose:
        print(f"Loaded {len(data)} items from {path}")
    return data
