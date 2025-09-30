"""Streamlit Dashboard for ML Predictions with Explanations.

This app loads a trained model (via dsworkflows PredictService), accepts user input
for prediction (single or batch), and visualizes results: prediction, probabilities,
feature importances/SHAP, and helpful charts.

Run locally:
    streamlit run dashboards/streamlit_app/app.py

Dependencies:
    - streamlit, pandas, numpy, plotly, shap (optional for visuals), scikit-learn
    - The dsworkflows package from this repo (installed via poetry/pip)

Author: Gabriel Demetrios Lafis (@galafis)
Date: 2025-09-30
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Try to import PredictService from repo package
try:
    from dsworkflows.models.predict import PredictService
except ImportError:
    # Fallback when running app directly without installing package
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))
    from dsworkflows.models.predict import PredictService  # type: ignore

DEFAULT_MODEL_PATH = os.getenv(
    "MODEL_PATH",
    str(Path(__file__).resolve().parents[2] / "models" / "trained_model.pkl")
)

# ===== Helper functions =====
def _ensure_dataframe(
    data: Any,
    feature_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Convert user-provided data to a pandas DataFrame.

    Accepts dict, list of dicts, or 2D list/array with optional feature_names.
    Raises ValueError for empty or mismatched inputs.
    """
    if data is None:
        raise ValueError("No data provided")

    if isinstance(data, dict):
        return pd.DataFrame([data])

    if isinstance(data, list):
        if len(data) == 0:
            raise ValueError("Empty list provided")
        # list of dicts
        if isinstance(data[0], dict):
            return pd.DataFrame(data)
        # list of lists/arrays
        arr = np.array(data)
        if feature_names is None:
            raise ValueError("feature_names are required when data is array-like")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != len(feature_names):
            raise ValueError("feature_names length must match number of columns in data")
        return pd.DataFrame(arr, columns=feature_names)

    raise ValueError("Unsupported data type. Use dict, list of dicts, or 2D list.")


def _shap_like_bar(explanation: Dict[str, float], title: str = "Feature importance"):
    """Create a horizontal bar chart for feature importances/SHAP values."""
    if not explanation:
        return None
    df_exp = (
        pd.Series(explanation, name="importance")
        .sort_values(ascending=False)
        .to_frame()
    )
    df_exp["feature"] = df_exp.index
    fig = px.bar(
        df_exp.iloc[::-1],  # plot from smallest to largest vertically
        x="importance",
        y="feature",
        orientation="h",
        title=title,
        text="importance",
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(yaxis_title="Feature", xaxis_title="Importance/SHAP")
    return fig


def _proba_plot(probabilities: Dict[str, float] | None, title: str = "Probabilities"):
    """Bar chart for class probabilities (binary/multiclass)."""
    if not probabilities:
        return None
    dfp = pd.DataFrame({"class": list(probabilities.keys()), "proba": list(probabilities.values())})
    fig = px.bar(dfp, x="class", y="proba", title=title, text="proba")
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_yaxes(range=[0, 1])
    return fig


def _load_predict_service(model_path: str) -> PredictService:
    """Instantiate PredictService with the given model path, raising helpful errors."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return PredictService(model_path)


# ===== Sidebar: Configuration =====
st.set_page_config(page_title="ML Prediction Dashboard", page_icon="🤖", layout="wide")
st.sidebar.title("⚙️ Configuração / Settings")

# Model selection
model_path = st.sidebar.text_input("Model path", value=DEFAULT_MODEL_PATH)
load_model_btn = st.sidebar.button("Load/Reload model")

# Prediction options
return_proba = st.sidebar.checkbox("Return probabilities", value=True)
explain = st.sidebar.checkbox("Include explanations (SHAP/importance)", value=True)

st.title("🤖 ML Prediction Dashboard")
st.caption(
    "Interactive app to run ML predictions, view probabilities and explainability charts."
)

# ===== Main: Data input =====
with st.expander("📥 Input data", expanded=True):
    st.write("Provide input as a JSON dict for single prediction, a list of dicts for batch, or a 2D list with feature_names.")

    example_dict = {
        "feature_1": 2.5,
        "feature_2": 1.8,
        "feature_3": 0.7,
        "feature_4": 3.2,
    }

    tab1, tab2, tab3 = st.tabs(["Single (dict)", "Batch (list of dicts)", "Array + feature_names"]) 

    with tab1:
        single_json = st.text_area(
            "JSON input (dict)",
            value=json.dumps(example_dict, indent=2),
            height=180,
        )

    with tab2:
        batch_json = st.text_area(
            "JSON input (list of dicts)",
            value=json.dumps([example_dict, {"feature_1": 1.1, "feature_2": 0.5, "feature_3": 2.2, "feature_4": 0.9}], indent=2),
            height=180,
        )

    with tab3:
        array_json = st.text_area(
            "JSON input (2D list)",
            value=json.dumps([[2.5, 1.8, 0.7, 3.2]], indent=2),
            height=120,
        )
        feature_names_text = st.text_input(
            "feature_names (comma-separated)", value="feature_1,feature_2,feature_3,feature_4"
        )
        feature_names = [f.strip() for f in feature_names_text.split(",") if f.strip()]

    input_mode = st.radio(
        "Select input mode",
        options=["dict", "list_of_dicts", "array"],
        horizontal=True,
        index=0,
    )

    # Parse user input
    parsed_data: Any = None
    parsed_feature_names: Optional[List[str]] = None
    parse_error: Optional[str] = None

    try:
        if input_mode == "dict":
            parsed_data = json.loads(single_json or "{}")
        elif input_mode == "list_of_dicts":
            parsed_data = json.loads(batch_json or "[]")
        else:
            parsed_data = json.loads(array_json or "[]")
            parsed_feature_names = feature_names
        # Validate/convert to DataFrame for preview
        df_preview = _ensure_dataframe(parsed_data, parsed_feature_names)
    except Exception as e:
        parse_error = str(e)
        df_preview = None

    if parse_error:
        st.error(f"Input parsing error: {parse_error}")
    else:
        st.success("Input parsed successfully.")
        if df_preview is not None:
            st.dataframe(df_preview.head(), use_container_width=True)

# ===== Load model =====
model_ok = False
predict_service: Optional[PredictService] = None

if load_model_btn:
    with st.spinner("Loading model..."):
        try:
            predict_service = _load_predict_service(model_path)
            st.session_state["predict_service"] = predict_service
            st.session_state["model_info"] = {
                "model_type": getattr(predict_service, "model_type", "Unknown"),
                "training_timestamp": getattr(predict_service, "training_timestamp", "Unknown"),
                "model_path": getattr(predict_service, "model_path", model_path),
            }
            st.success("Model loaded successfully.")
        except Exception as e:
            st.error(f"Failed to load model: {e}")

# Use previously loaded model from session, if any
predict_service = st.session_state.get("predict_service")  # type: ignore[assignment]
model_info = st.session_state.get("model_info")
model_ok = predict_service is not None

with st.expander("🧠 Model info", expanded=True):
    if model_ok and model_info:
        st.json(model_info)
    else:
        st.info("Model not loaded yet. Set the model path and click Load/Reload model.")

# ===== Predict =====
st.header("🔮 Prediction")
col_run, col_opts = st.columns([1, 2])
with col_run:
    run_btn = st.button("Run prediction", type="primary", use_container_width=True)
with col_opts:
    st.write("Options:")
    st.write(f"Return probabilities: {return_proba}")
    st.write(f"Explain: {explain}")

if run_btn:
    if not model_ok:
        st.warning("Please load the model first.")
    elif df_preview is None or df_preview.empty:
        st.warning("Please provide valid input data.")
    else:
        with st.spinner("Running inference..."):
            try:
                # Determine single vs batch based on provided structure
                is_batch = isinstance(parsed_data, list)

                if is_batch:
                    result = predict_service.predict_batch(  # type: ignore[union-attr]
                        X=parsed_data,
                        feature_names=parsed_feature_names,
                        return_proba=return_proba,
                        explain=explain,
                    )
                    predictions = result.get("predictions")
                    probabilities = result.get("probabilities")
                    explanations = result.get("explanations")

                    st.subheader("Results (batch)")
                    res_df = pd.DataFrame({"prediction": predictions})
                    if probabilities:
                        # probabilities could be list of dicts
                        prob_df = pd.DataFrame(probabilities)
                        res_df = pd.concat([res_df, prob_df], axis=1)
                    st.dataframe(res_df, use_container_width=True)

                    # Visuals for first sample
                    if return_proba and probabilities:
                        figp = _proba_plot(probabilities[0], title="Probabilities (first sample)")
                        if figp:
                            st.plotly_chart(figp, use_container_width=True)
                    if explain and explanations:
                        figi = _shap_like_bar(explanations[0], title="Feature importance (first sample)")
                        if figi:
                            st.plotly_chart(figi, use_container_width=True)

                else:
                    result = predict_service.predict_one(  # type: ignore[union-attr]
                        x=parsed_data,
                        feature_names=parsed_feature_names,
                        return_proba=return_proba,
                        explain=explain,
                    )
                    prediction = result.get("prediction")
                    probabilities = result.get("probabilities")
                    explanation = result.get("explanation")

                    st.subheader("Result (single)")
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        st.metric(label="Prediction", value=str(prediction))
                    with c2:
                        st.json({"prediction": prediction})

                    if return_proba and probabilities:
                        figp = _proba_plot(probabilities, title="Probabilities")
                        if figp:
                            st.plotly_chart(figp, use_container_width=True)
                    if explain and explanation:
                        figi = _shap_like_bar(explanation, title="Feature importance")
                        if figi:
                            st.plotly_chart(figi, use_container_width=True)

                st.success("Inference completed.")

            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ===== Synthetic example generator =====
st.header("🧪 Synthetic example")
with st.expander("Generate synthetic classification-like features", expanded=False):
    st.write("Quickly create a synthetic dataset for testing the dashboard.")
    n_samples = st.number_input("Number of samples", min_value=1, max_value=1000, value=5, step=1)
    rng_seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=42, step=1)
    gen_btn = st.button("Generate data")
    if gen_btn:
        rng = np.random.default_rng(int(rng_seed))
        synth = pd.DataFrame({
            "feature_1": rng.normal(0, 1, n_samples).round(3),
            "feature_2": rng.uniform(-1, 1, n_samples).round(3),
            "feature_3": rng.exponential(1.0, n_samples).round(3),
            "feature_4": rng.normal(2, 0.5, n_samples).round(3),
        })
        st.dataframe(synth, use_container_width=True)
        st.info("Copy to the input area as list of dicts or 2D array.")

# ===== Footer =====
st.caption(
    "Tip: Use the sidebar to load a model. Then paste your input and run predictions."
)
