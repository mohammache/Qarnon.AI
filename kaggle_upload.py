#!/usr/bin/env python3
"""
Upload MoroccoTone dataset to Kaggle.

Setup (one-time):
  1. Go to https://www.kaggle.com/settings → Account → API → Create New Token
  2. Save the downloaded kaggle.json to ~/.kaggle/kaggle.json
  3. chmod 600 ~/.kaggle/kaggle.json

Then run:
  python kaggle_upload.py
"""

import json
import os
import subprocess
from pathlib import Path

KAGGLE_USERNAME = input("Your Kaggle username: ").strip()
DATASET_SLUG    = "moroccoone-darija-instruction-dataset"
DATASET_TITLE   = "MoroccoTone Darija Instruction Dataset"
DATA_DIR        = Path("dataset")
UPLOAD_DIR      = Path("kaggle_upload_staging")

# Files to upload
FILES = ["train.jsonl", "val.jsonl", "dataset_stats.json"]

# Build staging dir with metadata
UPLOAD_DIR.mkdir(exist_ok=True)
for fname in FILES:
    src = DATA_DIR / fname
    dst = UPLOAD_DIR / fname
    if src.exists():
        dst.write_bytes(src.read_bytes())
        print(f"  staged: {fname}")

# Write dataset-metadata.json (required by Kaggle API)
metadata = {
    "title": DATASET_TITLE,
    "id": f"{KAGGLE_USERNAME}/{DATASET_SLUG}",
    "licenses": [{"name": "CC-BY-4.0"}],
    "resources": [
        {"path": f, "description": f.replace(".jsonl", " split").replace("_", " ")}
        for f in FILES
    ],
}
meta_path = UPLOAD_DIR / "dataset-metadata.json"
meta_path.write_text(json.dumps(metadata, indent=2))
print(f"\nMetadata written: {meta_path}")

# Check kaggle.json
kaggle_cfg = Path.home() / ".kaggle" / "kaggle.json"
if not kaggle_cfg.exists():
    print("\n⚠ No ~/.kaggle/kaggle.json found.")
    print("  Download from: https://www.kaggle.com/settings → API → Create New Token")
    exit(1)
os.chmod(kaggle_cfg, 0o600)

# Upload
print("\nUploading to Kaggle...")
result = subprocess.run(
    ["kaggle", "datasets", "create", "-p", str(UPLOAD_DIR), "--dir-mode", "zip"],
    capture_output=True, text=True
)
print(result.stdout)
if result.returncode != 0:
    print("ERROR:", result.stderr)
    # If dataset exists, update instead
    if "already exists" in result.stderr or "409" in result.stderr:
        print("Dataset exists — updating version...")
        result2 = subprocess.run(
            ["kaggle", "datasets", "version", "-p", str(UPLOAD_DIR),
             "-m", "Updated dataset", "--dir-mode", "zip"],
            capture_output=True, text=True
        )
        print(result2.stdout)
        if result2.returncode != 0:
            print("ERROR:", result2.stderr)
        else:
            print(f"\nDataset URL: https://www.kaggle.com/datasets/{KAGGLE_USERNAME}/{DATASET_SLUG}")
else:
    print(f"\nDataset URL: https://www.kaggle.com/datasets/{KAGGLE_USERNAME}/{DATASET_SLUG}")
