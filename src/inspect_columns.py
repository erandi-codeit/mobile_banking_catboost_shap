"""Utility: print dataset columns to help choose a target column."""

import argparse
from src.utils import read_dataset, clean_columns, drop_all_empty_columns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to dataset (.xlsx/.xls/.csv)")
    args = parser.parse_args()

    df = read_dataset(args.data)
    df = clean_columns(df)
    df = drop_all_empty_columns(df)

    print("Columns:\n")
    for c in df.columns:
        print(f"- {c}")

    print("\nRow count:", len(df))
    print("Column count:", len(df.columns))


if __name__ == "__main__":
    main()
