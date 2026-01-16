# src/train_catboost.py
"""
Train CatBoost + SHAP for the Sri Lanka Mobile Banking survey dataset.

Fixes:
1) Target is NEVER passed through numeric conversion / fill_missing logic.
2) Robust split retries seeds to avoid single-class training set.
3) Clear prints + saves artifacts to ./models
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import joblib
import pandas as pd
import shap

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from catboost import CatBoostClassifier, Pool

from src.utils import (
    read_dataset,
    clean_columns,
    infer_categorical_columns,
    fill_missing,
    split_xy,
    save_metadata,
)


def _safe_three_way_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    val_size: float,
    base_random_state: int,
    max_tries: int = 80,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Two-stage split (train_full/test then train/val) with safety checks:
    - Train MUST contain at least 2 unique classes (prevents CatBoost single-class error)
    - Retries different seeds if needed
    """
    y = y.astype("object").astype(str)

    # Drop missing/blank targets to avoid "nan" class
    y_clean = y.replace({"nan": None, "None": None, "": None}).copy()
    mask = y_clean.notna()
    X = X.loc[mask].reset_index(drop=True)
    y = y_clean.loc[mask].reset_index(drop=True).astype(str)

    classes = sorted(y.unique().tolist())
    if len(classes) < 2:
        raise ValueError(
            f"Target has only {len(classes)} unique class(es): {classes}. "
            "Choose a target with at least 2 classes."
        )

    for i in range(max_tries):
        rs = base_random_state + i

        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=rs,
            stratify=y,
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full,
            test_size=val_size,
            random_state=rs,
            stratify=y_train_full,
        )

        if y_train.nunique() >= 2:
            return X_train, X_val, X_test, y_train, y_val, y_test

    raise RuntimeError(
        "Could not create a split with >=2 classes in training after many retries. "
        "Try smaller --test_size/--val_size (e.g., val_size=0.1) or use more data."
    )

def _extract_categorical_options(X: pd.DataFrame, cat_cols: List[str], max_options: int = 50) -> dict:
    opts = {}
    for c in cat_cols:
        if c in X.columns:
            values = (
                X[c].astype(str)
                .replace({"nan": "Unknown", "None": "Unknown", "": "Unknown"})
                .fillna("Unknown")
                .unique()
                .tolist()
            )
            # sort for stable UI
            values = sorted(set(values))
            # cap (safety for high-cardinality)
            opts[c] = values[:max_options]
    return opts


def _extract_numeric_stats(X: pd.DataFrame, feature_names: List[str], cat_cols: List[str]) -> dict:
    stats = {}
    cat_set = set(cat_cols)
    for c in feature_names:
        if c in cat_set:
            continue
        s = pd.to_numeric(X[c], errors="coerce")
        if s.notna().sum() == 0:
            continue
        stats[c] = {
            "min": float(s.min()),
            "max": float(s.max()),
            "median": float(s.median()),
        }
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to survey dataset (.xlsx or .csv)")
    parser.add_argument("--target", type=str, required=True, help="Target column name (classification)")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)

    # CatBoost params
    parser.add_argument("--iterations", type=int, default=800)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=0.05)

    args = parser.parse_args()

    df = read_dataset(args.data)
    df = clean_columns(df)
    df = df.dropna(axis=1, how="all")

    # Split X/y first
    X, y = split_xy(df, args.target)

    # ✅ Preprocess X only (never touch y with numeric conversion)
    cat_cols: List[str] = infer_categorical_columns(df, args.target)

    Xy = pd.concat([X, y], axis=1)
    # Fill missing in X columns only, using the existing util
    X_filled = fill_missing(Xy.drop(columns=[args.target]), cat_cols)
    y_clean = Xy[args.target].astype("object")

    # Build final X/y
    X = X_filled
    # Keep Yes/No as strings; drop empty targets later in splitter
    y = y_clean.astype(str)

    # Robust split
    X_train, X_val, X_test, y_train, y_val, y_test = _safe_three_way_split(
        X=X,
        y=y,
        test_size=args.test_size,
        val_size=args.val_size,
        base_random_state=args.random_state,
        max_tries=120,
    )

    feature_names = list(X.columns)
    cat_feature_indices = [feature_names.index(c) for c in cat_cols if c in feature_names]

    train_pool = Pool(X_train, y_train, cat_features=cat_feature_indices)
    val_pool = Pool(X_val, y_val, cat_features=cat_feature_indices)
    test_pool = Pool(X_test, y_test, cat_features=cat_feature_indices)

    loss_function = "MultiClass" if y_train.nunique() > 2 else "Logloss"

    model = CatBoostClassifier(
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        loss_function=loss_function,
        eval_metric="Accuracy",
        random_seed=args.random_state,
        verbose=100,
    )

    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    preds = model.predict(test_pool).astype(str).reshape(-1)

    print("\n=== Target Distribution (Train/Val/Test) ===")
    print("Train:\n", y_train.value_counts())
    print("\nVal:\n", y_val.value_counts())
    print("\nTest:\n", y_test.value_counts())

    print("\n=== Classification Report (Test) ===")
    print(classification_report(y_test.astype(str), preds))

    print("\n=== Confusion Matrix (Test) ===")
    print(confusion_matrix(y_test.astype(str), preds))

    # Save artifacts
    Path("models").mkdir(exist_ok=True)
    model_path = "models/catboost_model.cbm"
    model.save_model(model_path)

    # SHAP explainer
    #background = X_train.sample(min(300, len(X_train)), random_state=args.random_state)
    #explainer = shap.TreeExplainer(model, data=background)
    #joblib.dump(explainer, "models/shap_explainer.pkl")

    # SHAP for CatBoost with categorical splits:
    # - Do NOT pass background data
    # - Use feature_perturbation="tree_path_dependent"
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    joblib.dump(explainer, "models/shap_explainer.pkl")
    
    categorical_options = _extract_categorical_options(pd.concat([X_train, X_val, X_test]), cat_cols)
    numeric_stats = _extract_numeric_stats(pd.concat([X_train, X_val, X_test]), feature_names, cat_cols)

    
    metadata = {
        "target_col": args.target,
        "feature_names": feature_names,
        "categorical_columns": cat_cols,
        "cat_feature_indices": cat_feature_indices,
        "classes": sorted(list(pd.Series(y_train).unique())),
        "data_path_used": str(args.data),
        "split": {"test_size": args.test_size, "val_size": args.val_size, "random_state": args.random_state},
        "note": "Target column excluded from numeric coercion; robust split retries seeds to avoid single-class train.",
        "categorical_options": categorical_options,
        "numeric_stats": numeric_stats,
    }
    save_metadata("models/metadata.json", metadata)

    print(f"\nSaved model to: {model_path}")
    print("Saved SHAP explainer to: models/shap_explainer.pkl")
    print("Saved metadata to: models/metadata.json")


if __name__ == "__main__":
    main()

