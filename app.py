# app.py (UPDATED: Model Info section added + Local SHAP in new row + Age int + Beeswarm explanation)
import json
from pathlib import Path

import joblib
import pandas as pd
import shap
import streamlit as st
import matplotlib.pyplot as plt

from catboost import CatBoostClassifier, Pool


st.set_page_config(
    page_title="Predicting and Explaining Mobile Banking Adoption in Sri Lanka",
    layout="wide",
)


@st.cache_resource
def load_artifacts():
    model = CatBoostClassifier()
    model.load_model("models/catboost_model.cbm")

    explainer = joblib.load("models/shap_explainer.pkl")

    with open("models/metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    return model, explainer, meta


def _is_positive_label(label) -> bool:
    s = str(label).strip().lower()
    return s in {"yes", "y", "true", "1"}


def build_input_form(meta):
    st.sidebar.header("Input Features")

    feature_names = meta["feature_names"]
    cat_cols = set(meta.get("categorical_columns", []))
    cat_options = meta.get("categorical_options", {})
    numeric_stats = meta.get("numeric_stats", {})

    inputs = {}

    for f in feature_names:
        if f in cat_cols:
            options = cat_options.get(f, [])
            if "Unknown" not in options:
                options = options + ["Unknown"]

            if len(options) > 1:
                # avoid defaulting to Unknown where possible
                default_idx = 0
                if options[0] == "Unknown" and len(options) > 1:
                    default_idx = 1
                inputs[f] = st.sidebar.selectbox(f, options, index=default_idx)
            else:
                inputs[f] = st.sidebar.text_input(f, value="")
        else:
            stats = numeric_stats.get(f)
            if stats:
                vmin, vmax, vmed = stats["min"], stats["max"], stats["median"]

                # ✅ Age as integer (no floats)
                if f.strip().lower() == "age":
                    inputs[f] = st.sidebar.number_input(
                        f,
                        min_value=int(vmin),
                        max_value=int(vmax),
                        value=int(round(vmed)),
                        step=1,
                    )
                else:
                    # other numeric features (Likert etc.)
                    if vmin == vmax:
                        inputs[f] = st.sidebar.number_input(f, value=int(vmed))
                    else:
                        inputs[f] = st.sidebar.slider(
                            f,
                            min_value=int(vmin),
                            max_value=int(vmax),
                            value=int(vmed),
                        )
            else:
                inputs[f] = st.sidebar.number_input(f, value=0.0)

    return inputs


def _cat_idx_from_meta(meta: dict):
    cat_idx = meta.get("cat_feature_indices", [])
    # Safety: ensure list[int]
    if isinstance(cat_idx, dict):
        cat_idx = sorted(list(cat_idx.values()))
    if not isinstance(cat_idx, list):
        cat_idx = list(cat_idx)
    return cat_idx


def show_local_explanation(explainer, X_one: pd.DataFrame, meta: dict):
    cat_idx = _cat_idx_from_meta(meta)

    pool = Pool(X_one, cat_features=cat_idx)
    shap_values = explainer.shap_values(pool)

    # Binary vs multiclass handling
    if isinstance(shap_values, list) and len(shap_values) == 2:
        sv = shap_values[1]  # positive class (Yes)
        base_value = explainer.expected_value[1]
    else:
        sv = shap_values
        base_value = explainer.expected_value

    exp = shap.Explanation(
        values=sv[0],
        base_values=base_value,
        data=X_one.iloc[0],
        feature_names=X_one.columns.tolist(),
    )

    plt.figure()
    shap.plots.waterfall(exp, show=False)
    st.pyplot(plt.gcf(), clear_figure=True)


def show_global_explanation(explainer, df: pd.DataFrame, meta: dict):
    cat_idx = _cat_idx_from_meta(meta)

    pool = Pool(df, cat_features=cat_idx)
    shap_values = explainer.shap_values(pool)

    # Binary classification: pick positive class (Yes)
    if isinstance(shap_values, list) and len(shap_values) == 2:
        sv = shap_values[1]
    else:
        sv = shap_values

    plt.figure()
    shap.summary_plot(sv, df, show=False)  # beeswarm by default
    st.pyplot(plt.gcf(), clear_figure=True)

def pretty_label(label: str) -> str:
    s = str(label).strip().lower()
    if s in {"y", "yes", "1", "true"}:
        return "Yes"
    if s in {"n", "no", "0", "false"}:
        return "No"
    return str(label)

def format_single_record_inline(X_one: pd.DataFrame) -> str:
    parts = []
    row = X_one.iloc[0]

    for col, val in row.items():
        parts.append(f"{col}: {val}")

    return " | ".join(parts)


def main():
    st.title("Predicting and Explaining Mobile Banking Adoption in Sri Lanka")
    st.caption("Sri Lanka Mobile Banking Survey — CatBoost Classification + SHAP Explainability")

    st.markdown(
        """
    <style>
    /* Smaller metric labels and values */
    div[data-testid="stMetricLabel"] {
        font-size: 0.75rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1rem;
    }
    </style>
        """,
        unsafe_allow_html=True,
    )
    if not Path("models/catboost_model.cbm").exists():
        st.error("Model artifacts not found. Train the model first (models/*.cbm, *.pkl, metadata.json).")
        st.stop()

    model, explainer, meta = load_artifacts()

    # Sidebar inputs -> model input row
    inputs = build_input_form(meta)
    X_one = pd.DataFrame([inputs], columns=meta["feature_names"])
   

    # =========================
    # Model Info / How it works
    # =========================
    target_col = meta.get("target_col", "Uses Mobile Banking?")
    classes = meta.get("classes", list(getattr(model, "classes_", ["Yes", "No"])))
    cat_cols = meta.get("categorical_columns", [])


    

    # =========================
    # ROW 1: Prediction
    # =========================
    st.subheader("🔮 Prediction")
    
    st.subheader("🧾 Prediction Input Summary")
    st.caption("Single input record used for prediction")

    summary = format_single_record_inline(X_one)
    st.markdown(f"`{summary}`")

    if st.button("Predict"):
        pred = model.predict(X_one)[0][0]
        raw_pred = pretty_label(pred)
        st.markdown(f"### Predicted Outcome: **{raw_pred}**")

        probs = model.predict_proba(X_one)[0]
        class_probs = dict(zip(model.classes_, probs))
        pred_prob = float(class_probs.get(pred, max(probs)))
        pred_prob_pct = round(pred_prob * 100, 1)

        if _is_positive_label(pred):
            msg = (
                "Based on the entered profile, the model predicts that "
                f"**this user will adopt mobile banking** with a probability of **{pred_prob_pct}%**."
            )
            st.success(msg)
        else:
            msg = (
                "Based on the entered profile, the model predicts that "
                f"**this user will not adopt mobile banking** with a probability of **{pred_prob_pct}%**."
            )
            st.warning(msg)

        probs_df = (
            pd.DataFrame({"Class": model.classes_, "Probability": probs})
            .sort_values("Probability", ascending=False)
            .reset_index(drop=True)
        )
        st.write("Class Probabilities")
        st.dataframe(probs_df, use_container_width=True)

    st.markdown("---")

    # =========================
    # ROW 2: Local SHAP (NEW ROW)
    # =========================
    st.subheader("🧠 Local Explanation (SHAP Waterfall)")
    st.caption("Explains *why* the model predicted Yes/No for the single profile entered in the sidebar.")

    if st.button("Explain this prediction"):
        show_local_explanation(explainer, X_one, meta)

    st.markdown("---")

    # =========================
    # Global SHAP
    # =========================
    st.subheader("🌍 Global Explanation (SHAP Beeswarm)")
    st.caption("Upload a sample file with the SAME feature columns used for training (target not required).")

    with st.expander("📘 How to read the SHAP Beeswarm plot"):
        st.markdown("""
**What this plot shows (Global Explanation)**  
- Each **row** is a feature, ordered from most important (top) to least important (bottom).  
- Each **dot** is one user record from the uploaded sample.  
- The **x-axis** shows the impact on the model prediction:
  - **Right** → pushes prediction toward **Adoption (Yes)**  
  - **Left** → pushes prediction toward **Non-adoption (No)**  
- **Red dots** represent **high feature values** and **blue dots** represent **low feature values**.  
- Features at the top influence mobile banking adoption **the most overall**.

✅ Use this plot to explain **what drives adoption across the population**, not a single user.
""")

    uploaded = st.file_uploader("Upload CSV or XLSX", type=["csv", "xlsx", "xls"])

    if uploaded is not None:
        if uploaded.name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded)
        else:
            df = pd.read_csv(uploaded)

        df = df.copy()

        # Keep only training features, in the correct order
        df = df[[c for c in meta["feature_names"] if c in df.columns]]

        # Fill missing (safe default)
        df = df.fillna("Unknown")

        # Sample for performance
        df_s = df.sample(min(300, len(df)), random_state=42)

        show_global_explanation(explainer, df_s, meta)


if __name__ == "__main__":
    main()


