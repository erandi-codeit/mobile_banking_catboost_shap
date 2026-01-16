import argparse
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import shap

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from catboost import CatBoostClassifier, Pool

from src.utils import (
    read_dataset,
    clean_columns,
    drop_all_empty_columns,
    infer_categorical_columns,
    fill_missing,
    split_xy,
    save_json,
)


def _safe_str_labels(y: pd.Series) -> pd.Series:
    # Survey targets are typically categorical; store as strings for consistent I/O.
    return y.astype("object").astype(str)


def main():
    parser = argparse.ArgumentParser(description="Train CatBoost classifier + build SHAP explainer")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset (.xlsx/.xls/.csv)")
    parser.add_argument("--target", type=str, required=True, help="Target column name")

    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)

    parser.add_argument("--iterations", type=int, default=800)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--l2_leaf_reg", type=float, default=3.0)

    parser.add_argument("--background_rows", type=int, default=300, help="Background sample size for SHAP")
    parser.add_argument("--max_unique_for_cat", type=int, default=30)

    args = parser.parse_args()

    df = read_dataset(args.data)
    df = clean_columns(df)
    df = drop_all_empty_columns(df)

    # Optional: drop rows that are completely empty
    df = df.dropna(axis=0, how="all")

    # Split and infer types
    X_raw, y_raw = split_xy(df, args.target)

    cat_cols: List[str] = infer_categorical_columns(df, args.target, max_unique_for_cat=args.max_unique_for_cat)

    df2 = pd.concat([X_raw, y_raw], axis=1)
    df2 = fill_missing(df2, cat_cols)
    X, y = split_xy(df2, args.target)
    y = _safe_str_labels(y)

    # Stratify only if number of classes is not too huge
    strat = y if y.nunique() <= 20 else None

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=strat,
    )

    strat2 = y_train_full if y_train_full.nunique() <= 20 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=args.val_size,
        random_state=args.random_state,
        stratify=strat2,
    )

    feature_names = list(X.columns)
    cat_feature_indices = [feature_names.index(c) for c in cat_cols if c in feature_names]

    train_pool = Pool(X_train, y_train, cat_features=cat_feature_indices)
    val_pool = Pool(X_val, y_val, cat_features=cat_feature_indices)
    test_pool = Pool(X_test, y_test, cat_features=cat_feature_indices)

    loss_function = "MultiClass" if y.nunique() > 2 else "Logloss"

    model = CatBoostClassifier(
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        l2_leaf_reg=args.l2_leaf_reg,
        loss_function=loss_function,
        eval_metric="Accuracy",
        random_seed=args.random_state,
        verbose=100,
        allow_writing_files=False,
    )

    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    preds = model.predict(test_pool)
    preds = preds.astype(str).reshape(-1)

    print("\n=== Classification Report ===")
    print(classification_report(y_test.astype(str), preds))

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test.astype(str), preds))

    # Save artifacts
    Path("models").mkdir(exist_ok=True)
    model_path = "models/catboost_model.cbm"
    model.save_model(model_path)

    # SHAP TreeExplainer
    background = X_train.sample(min(args.background_rows, len(X_train)), random_state=args.random_state)
    explainer = shap.TreeExplainer(model, data=background)
    joblib.dump(explainer, "models/shap_explainer.pkl")

    metadata = {
        "target_col": args.target,
        "feature_names": feature_names,
        "categorical_columns": cat_cols,
        "cat_feature_indices": cat_feature_indices,
        "classes": list(getattr(model, "classes_", sorted(list(y.unique())))),
        "data_path_used": str(args.data),
        "notes": {
            "missing_values": "categorical->Unknown, numeric->median",
            "label_cast": "y cast to string",
        },
    }
    save_json("models/metadata.json", metadata)

    print("\nSaved:")
    print(f"- Model: {model_path}")
    print("- SHAP explainer: models/shap_explainer.pkl")
    print("- Metadata: models/metadata.json")


if __name__ == "__main__":
    main()
