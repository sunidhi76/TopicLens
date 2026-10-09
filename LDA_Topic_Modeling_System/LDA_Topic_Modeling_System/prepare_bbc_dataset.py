"""Prepare the real BBC News dataset for TopicLens.

This script converts the common BBC folder layout:
bbc/
  business/*.txt
  entertainment/*.txt
  politics/*.txt
  sport/*.txt
  tech/*.txt

into data/bbc_news.csv. Category labels are retained only as metadata; TopicLens
does not feed them to LDA.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def prepare(input_dir: Path, output_file: Path):
    rows = []
    for category_dir in sorted(input_dir.iterdir()):
        if not category_dir.is_dir():
            continue
        category = category_dir.name
        for path in sorted(category_dir.glob("*.txt")):
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                rows.append({
                    "text": text,
                    "category": category,
                    "filename": path.name,
                })
    if not rows:
        raise SystemExit("No .txt documents found in the expected category folders.")
    df = pd.DataFrame(rows)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Created {output_file} with {len(df)} documents.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/bbc", help="Extracted BBC dataset directory")
    parser.add_argument("--output", default="data/bbc_news.csv")
    args = parser.parse_args()
    prepare(Path(args.input), Path(args.output))
