# Sri Lanka Mobile Banking Survey — CatBoost + SHAP + Streamlit

This is a complete Python project to train a **CatBoost** model on a **tabular survey dataset**, evaluate it, and explain predictions using **SHAP**.

It also includes a **Streamlit** app : https://predictslmobilebankingadoption.streamlit.app/

## 1) Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Add your dataset

Put the Kaggle dataset file inside `data/`:

- `data/survey.xlsx` (recommended)
- or `data/survey.csv`

## 3) Inspect columns (to choose a target)

```bash
python3 -m src.inspect_columns --data data/survey.xlsx
```

Pick a suitable target column (examples):
- `Uses Mobile Banking`
- `Satisfaction Level`

## 4) Train CatBoost + SHAP

```bash
python3 -u -m src.train_catboost --data data/survey.xlsx --target "Uses Mobile Banking?" --test_size 0.15 --val_size 0.05


```

Artifacts will be saved to:
- `models/catboost_model.cbm`
- `models/shap_explainer.pkl`
- `models/metadata.json`

## 5) Run the Streamlit app

```bash
python -m streamlit run app.py
```

### App features
- Predict a class for user-entered features
- Show **local explanation** (SHAP waterfall)
- Upload a sample file to show **global explanation** (SHAP beeswarm)

## Notes
- Missing values are handled as: **categorical → "Unknown"**, **numeric → median**.
- Categorical columns are inferred from dtype (object/category) and small-unique-count heuristic.

