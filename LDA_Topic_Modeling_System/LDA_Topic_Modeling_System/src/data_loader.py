"""Dataset loading and validation utilities for TopicLens."""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pandas as pd


TEXT_COLUMN_CANDIDATES = (
    "text", "article", "content", "body", "document", "description", "news"
)
LABEL_COLUMN_CANDIDATES = ("category", "label", "class", "topic")


class DatasetError(ValueError):
    """Raised when a dataset cannot be used for LDA."""


def _find_text_column(columns) -> str | None:
    lowered = {str(c).strip().lower(): c for c in columns}
    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]
    return None


def _clean_frame(df: pd.DataFrame, source_name: str) -> tuple[pd.DataFrame, str | None]:
    if df is None or df.empty:
        raise DatasetError("The dataset is empty.")

    text_col = _find_text_column(df.columns)
    if text_col is None:
        raise DatasetError(
            "No text column was found. Use a column named text, article, content, "
            "body, document, description, or news."
        )

    out = df.copy()
    out["document_id"] = [f"DOC-{i+1:04d}" for i in range(len(out))]
    out["text"] = out[text_col].fillna("").astype(str).str.strip()
    out = out[out["text"].str.len() > 0].reset_index(drop=True)

    if out.empty:
        raise DatasetError("No non-empty text documents remain after validation.")

    label_col = next(
        (c for c in out.columns if str(c).strip().lower() in LABEL_COLUMN_CANDIDATES),
        None,
    )
    out["source"] = source_name
    keep = ["document_id", "text", "source"]
    if label_col is not None:
        out["label_metadata"] = out[label_col].fillna("").astype(str)
        keep.append("label_metadata")
    return out[keep], label_col


def load_bbc_csv(path: str | Path = "data/bbc_news.csv") -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise DatasetError(
            "BBC dataset not found at data/bbc_news.csv. Follow README.md to prepare it."
        )
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise DatasetError(f"Could not read the BBC CSV: {exc}") from exc
    out, _ = _clean_frame(df, "BBC News Dataset")
    if len(out) < 2:
        raise DatasetError("The BBC dataset needs at least 2 valid documents.")
    return out


def load_csv_bytes(data: bytes, source_name: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.BytesIO(data))
    except Exception as exc:
        raise DatasetError(f"Invalid CSV file: {exc}") from exc
    out, _ = _clean_frame(df, source_name)
    return out


def load_txt_bytes(data: bytes, source_name: str) -> pd.DataFrame:
    try:
        raw = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DatasetError("TXT must be UTF-8 encoded.") from exc

    # Blank-line separated paragraphs are treated as documents. If there are no
    # blank lines, each non-empty line becomes one document.
    blocks = [b.strip() for b in re.split(r"\n\s*\n", raw) if b.strip()]
    if len(blocks) <= 1:
        blocks = [line.strip() for line in raw.splitlines() if line.strip()]
    df = pd.DataFrame({"text": blocks})
    out, _ = _clean_frame(df, source_name)
    return out


def load_json_bytes(data: bytes, source_name: str) -> pd.DataFrame:
    try:
        payload: Any = json.loads(data.decode("utf-8-sig"))
    except Exception as exc:
        raise DatasetError(f"Invalid JSON file: {exc}") from exc

    if isinstance(payload, dict):
        for key in ("documents", "data", "articles", "records"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break

    if isinstance(payload, list) and payload and all(isinstance(x, str) for x in payload):
        df = pd.DataFrame({"text": payload})
    elif isinstance(payload, list) and payload and all(isinstance(x, dict) for x in payload):
        df = pd.DataFrame(payload)
    else:
        raise DatasetError(
            "JSON must contain a list of strings or a list of objects with a text-like column."
        )

    out, _ = _clean_frame(df, source_name)
    return out


def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    name = getattr(uploaded_file, "name", "uploaded_dataset")
    suffix = Path(name).suffix.lower()
    data = uploaded_file.getvalue()
    if not data:
        raise DatasetError("The uploaded file is empty.")
    if suffix == ".csv":
        return load_csv_bytes(data, f"Uploaded: {name}")
    if suffix == ".txt":
        return load_txt_bytes(data, f"Uploaded: {name}")
    if suffix == ".json":
        return load_json_bytes(data, f"Uploaded: {name}")
    raise DatasetError("Unsupported file type. Please upload CSV, TXT, or JSON.")


def dataset_summary(df: pd.DataFrame) -> dict[str, Any]:
    lengths = df["text"].str.split().str.len()
    return {
        "documents": int(len(df)),
        "total_words": int(lengths.sum()),
        "average_length": float(lengths.mean()),
        "minimum_length": int(lengths.min()),
        "maximum_length": int(lengths.max()),
        "columns": int(len(df.columns)),
    }
