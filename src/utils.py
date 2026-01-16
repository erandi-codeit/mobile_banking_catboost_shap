# src/utils.py
import json
from pathlib import Path
from typing import List, Tuple

import pandas as pd


def read_dataset(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p.resolve()}")

    if p.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(p)
    elif p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    else:
        raise ValueError("Unsupported file type. Use .xlsx/.xls or .csv")


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        str(c).strip().replace("\n", " ").replace("\t", " ")
        for c in df.columns
    ]
    return df


def infer_categorical_columns(df: pd.DataFrame, target_col: str) -> List[str]:
    cat_cols = []
    for c in df.columns:
        if c == target_col:
            continue
        if df[c].dtype == "object" or str(df[c].dtype).startswith("category"):
            cat_cols.append(c)
    return cat_cols


def fill_missing(df: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if c in cat_cols:
            df[c] = df[c].astype("object").fillna("Unknown")
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].fillna(df[c].median())
    return df


def split_xy(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found")
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()
    return X, y


def save_metadata(path: str, metadata: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

