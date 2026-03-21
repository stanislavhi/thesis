"""
ARC-AGI Dataset Loader.

Downloads the ARC-AGI dataset from GitHub and provides utilities
to load and inspect individual tasks.
"""
import json
import os
import urllib.request
from typing import List, Optional
import numpy as np

# ARC-AGI public dataset URLs
ARC_REPO = "https://raw.githubusercontent.com/fchollet/ARC-AGI/master/data"
SPLITS = {
    "training": f"{ARC_REPO}/training",
    "evaluation": f"{ARC_REPO}/evaluation",
}

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/arc'))


def ensure_data_dir():
    """Create data directories if they don't exist."""
    for split in SPLITS:
        path = os.path.join(DATA_DIR, split)
        os.makedirs(path, exist_ok=True)


def download_task(task_id: str, split: str = "training") -> dict:
    """Download a single ARC task JSON from GitHub."""
    ensure_data_dir()
    local_path = os.path.join(DATA_DIR, split, f"{task_id}.json")

    if os.path.exists(local_path):
        with open(local_path) as f:
            return json.load(f)

    url = f"{SPLITS[split]}/{task_id}.json"
    try:
        urllib.request.urlretrieve(url, local_path)
        with open(local_path) as f:
            return json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to download task {task_id}: {e}") from e


def load_task(path: str) -> dict:
    """Load an ARC task from a local JSON file."""
    with open(path) as f:
        return json.load(f)


def list_local_tasks(split: str = "training") -> List[str]:
    """List all locally available task IDs."""
    path = os.path.join(DATA_DIR, split)
    if not os.path.exists(path):
        return []
    return sorted([f.replace(".json", "") for f in os.listdir(path) if f.endswith(".json")])


def download_sample_tasks(n: int = 20, split: str = "training") -> List[str]:
    """
    Download a sample of ARC tasks.
    Uses a curated list of known task IDs that represent different difficulty levels.
    """
    # A curated set of well-known ARC task IDs (from the training set)
    SAMPLE_IDS = [
        "007bbfb7", "00d62c1b", "017c7c7b", "025d127b", "045e512c",
        "0520fde7", "05f2a901", "06df4c85", "08ed6ac7", "09629e4f",
        "0962bcdd", "0a938d79", "0b148d64", "0ca9ddb6", "0d3d703e",
        "0e206a2e", "10fcaaa3", "11852cab", "1190e5a7", "12eac192",
    ][:n]

    downloaded = []
    for task_id in SAMPLE_IDS:
        try:
            download_task(task_id, split)
            downloaded.append(task_id)
        except Exception as e:
            print(f"  Warning: could not download {task_id}: {e}")

    return downloaded


def task_summary(task: dict) -> dict:
    """Get a summary of a task's structure."""
    train = task.get("train", [])
    test = task.get("test", [])

    return {
        "n_train": len(train),
        "n_test": len(test),
        "input_shapes": [np.array(ex["input"]).shape for ex in train],
        "output_shapes": [np.array(ex["output"]).shape for ex in train],
        "input_colors": list(set(
            c for ex in train
            for row in ex["input"]
            for c in row
        )),
        "output_colors": list(set(
            c for ex in train
            for row in ex["output"]
            for c in row
        )),
    }


def grid_to_string(grid) -> str:
    """Convert a grid (list of lists or numpy array) to a readable string."""
    if isinstance(grid, np.ndarray):
        grid = grid.tolist()
    COLOR_MAP = {
        0: "⬛", 1: "🟦", 2: "🟥", 3: "🟩", 4: "🟨",
        5: "⬜", 6: "🟪", 7: "🟧", 8: "🩵", 9: "🟫",
    }
    lines = []
    for row in grid:
        lines.append("".join(COLOR_MAP.get(c, "❓") for c in row))
    return "\n".join(lines)
