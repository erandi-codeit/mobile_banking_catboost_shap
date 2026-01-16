import json
from pathlib import Path
from typing import Tuple, List, Dict, Any

import pandas as pd


def read_dataset(path: str) -> pd.DataFrame:
    """Read CSV/XLSX dataset."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p.resolve()}")

    suf = p.suffix.lower()
    if suf in {".xlsx", ".xls"}:
        return pd.read_excel(p)
    if suf == ".csv":
        return pd.read_csv(p)
    raise ValueError("Unsupported file type. Use .xlsx/.xls or .csv")


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names."""
    df = df.copy()
    df.columns = [
        str(c)
        .strip()
        .replace("\n", " ")
        .replace("\t", " ")
        .replace("  ", " ")
        for c in df.columns
    ]
    return df


def drop_all_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(axis=1, how="all")


def infer_categorical_columns(df: pd.DataFrame, target_col: str, max_unique_for_cat: int = 30) -> List[str]:
    """Infer categorical columns using dtype and unique-count heuristic."""
    cat_cols: List[str] = []
    for c in df.columns:
        if c == target_col:
            continue
        if df[c].dtype == "object" or str(df[c].dtype).startswith("category"):
            cat_cols.append(c)
            continue
        # numeric-looking columns with very few unique values often represent categories
        try:
            nun = df[c].nunique(dropna=True)
            if nun <= max_unique_for_cat and nun > 1:
                # only treat as categorical if values are integer-like
                s = pd.to_numeric(df[c], errors="coerce")
                if (s.dropna() % 1 == 0).all():
                    cat_cols.append(c)
        except Exception:
            pass
    return sorted(list(set(cat_cols)))


def fill_missing(df: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
    """Fill missing values: categorical -> 'Unknown', numeric -> median."""
    df = df.copy()
    for c in df.columns:
        if c in cat_cols:
            df[c] = df[c].astype("object").fillna("Unknown")
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            med = df[c].median()
            df[c] = df[c].fillna(med)
    return df


def split_xy(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise KeyError(
            f"Target column '{target_col}' not found. Available columns: {list(df.columns)}"
        )
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()
    return X, y


def save_json(path: str, payload: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
