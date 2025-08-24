import streamlit as st
import pandas as pd
import numpy as np
import pickle
import pathlib
import time
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import statsmodels

# ===============================
# Page config
# ===============================
st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺", layout="wide")

# ===============================
# Load model & artifacts
# ===============================
@st.cache_resource
def load_artifacts():
    with open("model_pickle.pkl", "rb") as f:
        model = pickle.load(f)
    with open("feature_info_pickle.pkl", "rb") as f:
        feature_info = pickle.load(f)
    with open("metrics_pickle.pkl", "rb") as f:
        metrics = pickle.load(f)
    return model, feature_info, metrics

@st.cache_data
def load_data():
    data_path = pathlib.Path("data") / "diabetes.csv"
    if data_path.exists():
        return pd.read_csv(data_path)
    else:
        return pd.DataFrame()

model, feature_info, metrics = load_artifacts()
df = load_data()
target = metrics.get("target", "Outcome")

# ===============================
# Sidebar Navigation
# ===============================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["🏠 Overview", "🔍 Data Exploration", "📊 Visualisations", "🤖 Model Prediction", "📈 Model Performance"]
)

# ===============================
# 1) Overview
# ===============================
if page == "🏠 Overview":
    st.title("🩺 Diabetes Prediction App")
    st.markdown("""
    This application demonstrates a complete **Machine Learning workflow** for predicting the likelihood of diabetes.
    
    **Features:**
    - Interactive dataset exploration  
    - Multiple visualisations  
    - Real-time prediction with model confidence  
    - Performance evaluation and model comparison  
    """)
    st.subheader("Selected Model")
    st.write(f"**{metrics['test_metrics']['model']}**")
    st.json(metrics["test_metrics"])

# ===============================
# 2) Data Exploration
# ===============================
elif page == "🔍 Data Exploration":
    st.header("Dataset Overview")

    if df.empty:
        st.warning("Dataset not found. Please place `diabetes.csv` in the `data/` folder.")
    else:
        st.write(f"**Shape:** {df.shape}")
        st.write(f"**Columns:** {list(df.columns)}")
        st.write("**Data Types:**")
        st.write(df.dtypes)

        st.subheader("Sample Data")
        st.dataframe(df.head())

        # Interactive Filtering
        st.subheader("Filter Data")
        filters = {}
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            sel_range = st.slider(f"{col} range", min_val, max_val, (min_val, max_val))
            filters[col] = sel_range

        filtered_df = df.copy()
        for col, (low, high) in filters.items():
            filtered_df = filtered_df[(filtered_df[col] >= low) & (filtered_df[col] <= high)]

        st.write(f"Filtered Rows: {filtered_df.shape[0]}")
        st.dataframe(filtered_df)

# ===============================
# 3) Visualisations
# ===============================
elif page == "📊 Visualisations":
    st.header("Interactive Visualisations")

    if df.empty:
        st.warning("Dataset not found. Please place `diabetes.csv` in the `data/` folder.")
    else:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        color_col = st.selectbox("Color by:", [None] + list(df.columns), index=0)

        # Chart 1: Histogram
        st.subheader("Histogram")
        col_choice = st.selectbox("Select Feature", num_cols)
        fig_hist = px.histogram(df, x=col_choice, color=color_col)
        st.plotly_chart(fig_hist, use_container_width=True)

        # Chart 2: Scatter plot
        st.subheader("Scatter Plot")
        x_axis = st.selectbox("X Axis", num_cols, index=0)
        y_axis = st.selectbox("Y Axis", num_cols, index=min(1, len(num_cols)-1))
        fig_scatter = px.scatter(df, x=x_axis, y=y_axis, color=color_col, trendline="ols")
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Chart 3: Correlation Heatmap
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(df[num_cols].corr(), cmap="coolwarm", annot=False, ax=ax)
        st.pyplot(fig)

# ===============================
# 4) Model Prediction
# ===============================
elif page == "🤖 Model Prediction":
    st.header("Make a Prediction")

    with st.form("prediction_form"):
        input_data = {}
        for col, stats in feature_info.items():
            step_val = (stats["max"] - stats["min"]) / 100 if stats["max"] > stats["min"] else 0.1
            input_data[col] = st.number_input(
                col,
                min_value=stats["min"],
                max_value=stats["max"],
                value=stats["median"],
                step=step_val
            )
        submitted = st.form_submit_button("Predict")

    if submitted:
        X_new = pd.DataFrame([input_data])
        with st.spinner("Making prediction..."):
            time.sleep(1)
            prediction = model.predict(X_new)[0]
            proba = model.predict_proba(X_new)[0, 1] if hasattr(model, "predict_proba") else None

        st.success(f"Prediction: **{int(prediction)}** (0 = No Diabetes, 1 = Diabetes)")
        if proba is not None:
            st.info(f"Prediction Probability: **{proba:.2f}**")

# ===============================
# 5) Model Performance
# ===============================
elif page == "📈 Model Performance":
    st.header("Model Evaluation")

    st.subheader("Cross-Validation Results")
    st.json(metrics["cv_results"])

    st.subheader("Test Metrics")
    st.json(metrics["test_metrics"])

    cm_path = pathlib.Path("confusion_matrix.png")
    if cm_path.exists():
        st.image(str(cm_path), caption="Confusion Matrix", use_column_width=True)
    else:
        st.warning("Confusion matrix image not found.")
