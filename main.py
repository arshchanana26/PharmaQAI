# main.py
# Streamlit app: ML-based prediction of tablet quality attributes from manufacturing data
# Loads CSV from repository (no upload required), trains models, interactive plots + contour exploration.

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.inspection import permutation_importance

import plotly.express as px
import plotly.graph_objects as go


# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="PharmaQualityAI", layout="wide")

DATA_FILENAME = "Process (1).csv"  # keep exactly as committed in repo
DATA_PATH = Path(__file__).resolve().parent / DATA_FILENAME

KNOWN_CQA_TARGETS = [
    "Drug release average (%)",
    "Drug release min (%)",
    "Residual solvent",
    "Total impurities",
    "Impurity O",
    "Impurity L",
]

ID_COLS = {"batch", "code"}  # identifier columns (never used as targets)


# -----------------------------
# Data loading
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        st.error(f"Dataset not found at: {DATA_PATH}")
        st.stop()

    df = pd.read_csv(DATA_PATH, sep=";")

    # Normalize whitespace in column names (safe, helps avoid hidden issues)
    df.columns = [c.strip() for c in df.columns]

    return df


def detect_targets(df: pd.DataFrame) -> List[str]:
    # Prefer known CQAs if present
    present = [c for c in KNOWN_CQA_TARGETS if c in df.columns]

    # Add keyword-based targets (in case naming differs)
    keyword_hits: List[str] = []
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in ["impur", "release", "solvent", "dissol", "hard", "assay", "uniform"]):
            if c not in present and c not in ID_COLS:
                keyword_hits.append(c)

    # Fallback: numeric columns not identifiers
    numeric = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ID_COLS]

    ordered: List[str] = []
    for c in present + keyword_hits + numeric:
        if c not in ordered and c in df.columns:
            ordered.append(c)

    # Last safety: if still empty, allow any column
    if not ordered:
        ordered = [c for c in df.columns if c not in ID_COLS]

    return ordered


def feature_types(df: pd.DataFrame, target_col: str) -> Tuple[List[str], List[str]]:
    X = df.drop(columns=[target_col])
    cat = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype) == "category"]
    num = [c for c in X.columns if c not in cat]
    return num, cat


def build_preprocessor(num_cols: List[str], cat_cols: List[str], scale_numeric: bool) -> ColumnTransformer:
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(steps=num_steps)

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return []


# -----------------------------
# Modeling
# -----------------------------
def get_model(problem_mode: str, model_name: str, seed: int):
    if problem_mode == "regression":
        if model_name == "Random Forest":
            return RandomForestRegressor(n_estimators=400, random_state=seed, n_jobs=-1)
        if model_name == "ElasticNet":
            return ElasticNet(random_state=seed)
        raise ValueError("Unknown regression model")
    else:
        if model_name == "Random Forest":
            return RandomForestClassifier(
                n_estimators=400, random_state=seed, n_jobs=-1, class_weight="balanced"
            )
        if model_name == "Logistic Regression":
            return LogisticRegression(max_iter=5000, class_weight="balanced")
        raise ValueError("Unknown classification model")


@st.cache_data(show_spinner=False)
def prepare_xy(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    data = df.dropna(subset=[target_col]).copy()
    X = data.drop(columns=[target_col])
    y = data[target_col].copy()
    return X, y


@st.cache_resource(show_spinner=False)
def train_pipeline(
    df: pd.DataFrame,
    target_col: str,
    problem_mode: str,
    model_name: str,
    test_size: float,
    seed: int,
    scale_numeric: bool,
) -> Tuple[Pipeline, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X, y = prepare_xy(df, target_col)

    stratify = None
    if problem_mode == "classification" and y.nunique(dropna=True) <= 20:
        stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify
    )

    num_cols, cat_cols = feature_types(pd.concat([X, y], axis=1), target_col)
    pre = build_preprocessor(num_cols, cat_cols, scale_numeric=scale_numeric)
    model = get_model(problem_mode, model_name, seed)

    pipe = Pipeline(steps=[("preprocessor", pre), ("model", model)])
    pipe.fit(X_train, y_train)

    return pipe, X_train, X_test, y_train, y_test


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse,
    }


def classification_metrics(y_true: pd.Series, y_pred: np.ndarray, y_proba: Optional[np.ndarray]) -> Dict[str, float]:
    out = {"Accuracy": float(accuracy_score(y_true, y_pred))}
    if y_proba is not None and y_true.nunique() == 2:
        out["ROC AUC"] = float(roc_auc_score(y_true, y_proba))
    return out


# -----------------------------
# Plot helpers
# -----------------------------
def plot_correlation_heatmap(df: pd.DataFrame):
    num = df.select_dtypes(include=[np.number]).copy()
    if num.shape[1] < 2:
        st.info("Not enough numeric columns for correlation heatmap.")
        return
    corr = num.corr(numeric_only=True)
    fig = px.imshow(corr, aspect="auto", title="Correlation heatmap (numeric columns)")
    st.plotly_chart(fig, width="stretch")


def plot_feature_importance(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, problem_mode: str):
    pre = pipe.named_steps["preprocessor"]
    names = get_feature_names(pre)
    if not names:
        st.info("Could not extract feature names for importance.")
        return

    scoring = "r2" if problem_mode == "regression" else ("roc_auc" if y_test.nunique() == 2 else "accuracy")

    try:
        r = permutation_importance(
            pipe, X_test, y_test, n_repeats=12, random_state=0, n_jobs=-1, scoring=scoring
        )
    except Exception as e:
        st.warning(f"Permutation importance failed: {e}")
        return

    imp = (
        pd.DataFrame(
            {"feature": names, "importance_mean": r.importances_mean, "importance_std": r.importances_std}
        )
        .sort_values("importance_mean", ascending=False)
        .head(25)
    )

    fig = px.bar(
        imp.iloc[::-1],
        x="importance_mean",
        y="feature",
        orientation="h",
        error_x="importance_std",
        title="Permutation importance (top 25)",
    )
    st.plotly_chart(fig, width="stretch")


def plot_predictions(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, problem_mode: str):
    if problem_mode == "regression":
        y_pred = pipe.predict(X_test)
        met = regression_metrics(y_test, y_pred)

        c1, c2, c3 = st.columns(3)
        c1.metric("R2", f"{met['R2']:.4f}")
        c2.metric("MAE", f"{met['MAE']:.4f}")
        c3.metric("RMSE", f"{met['RMSE']:.4f}")

        dfp = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
        fig1 = px.scatter(dfp, x="Actual", y="Predicted", title="Actual vs Predicted")
        st.plotly_chart(fig1, width="stretch")

        resid = y_test.values - y_pred
        fig2 = px.scatter(
            x=y_pred,
            y=resid,
            labels={"x": "Predicted", "y": "Residual"},
            title="Residuals vs Predicted",
        )
        st.plotly_chart(fig2, width="stretch")

    else:
        y_pred = pipe.predict(X_test)
        y_proba = None
        if hasattr(pipe, "predict_proba") and y_test.nunique() == 2:
            y_proba = pipe.predict_proba(X_test)[:, 1]

        met = classification_metrics(y_test, y_pred, y_proba)
        cols = st.columns(len(met))
        for i, (k, v) in enumerate(met.items()):
            cols[i].metric(k, f"{v:.4f}")

        st.text("Classification report")
        st.text(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(cm, text_auto=True, title="Confusion matrix")
        st.plotly_chart(fig, width="stretch")

        if y_proba is not None:
            fig2 = px.histogram(
                pd.DataFrame({"p(out_of_spec)": y_proba}),
                x="p(out_of_spec)",
                nbins=30,
                title="Predicted probability distribution",
            )
            st.plotly_chart(fig2, width="stretch")


def contour_explorer(pipe: Pipeline, df: pd.DataFrame, target_col: str, problem_mode: str):
    data = df.dropna(subset=[target_col]).copy()
    X = data.drop(columns=[target_col])
    y = data[target_col]

    num_cols = list(X.select_dtypes(include=[np.number]).columns)
    if len(num_cols) < 2:
        st.info("Need at least two numeric input features for contour plots.")
        return

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        x_feat = st.selectbox("X feature", options=num_cols, index=0)
    with c2:
        y_feat = st.selectbox("Y feature", options=[c for c in num_cols if c != x_feat], index=0)
    with c3:
        grid_n = st.slider("Grid resolution", min_value=25, max_value=120, value=60, step=5)

    # Hold other features at median (numeric) / mode (categorical)
    fixed: Dict[str, object] = {}
    for col in X.columns:
        if col in (x_feat, y_feat):
            continue
        if X[col].dtype == "object" or str(X[col].dtype) == "category":
            fixed[col] = X[col].dropna().mode().iloc[0] if X[col].dropna().shape[0] else "unknown"
        else:
            fixed[col] = float(pd.to_numeric(X[col], errors="coerce").median())

    # grid ranges based on 1st and 99th percentiles
    x_vals = pd.to_numeric(X[x_feat], errors="coerce").dropna().values
    y_vals = pd.to_numeric(X[y_feat], errors="coerce").dropna().values
    x_min, x_max = np.nanpercentile(x_vals, [1, 99])
    y_min, y_max = np.nanpercentile(y_vals, [1, 99])

    gx = np.linspace(x_min, x_max, grid_n)
    gy = np.linspace(y_min, y_max, grid_n)
    XX, YY = np.meshgrid(gx, gy)

    grid = pd.DataFrame({x_feat: XX.ravel(), y_feat: YY.ravel()})
    for col, val in fixed.items():
        grid[col] = val

    if problem_mode == "regression":
        Z = pipe.predict(grid).reshape(YY.shape)
        z_title = f"Predicted {target_col}"
    else:
        if hasattr(pipe, "predict_proba") and y.nunique() == 2:
            Z = pipe.predict_proba(grid)[:, 1].reshape(YY.shape)
            z_title = "Predicted p(out_of_spec)"
        else:
            Z = pipe.predict(grid).reshape(YY.shape)
            z_title = "Predicted class"

    fig = go.Figure()
    fig.add_trace(go.Contour(x=gx, y=gy, z=Z, contours_coloring="heatmap", colorbar_title=z_title))

    # Overlay observed points (sample)
    sample_n = min(600, len(X))
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), size=sample_n, replace=False)
    fig.add_trace(
        go.Scatter(
            x=pd.to_numeric(X.iloc[idx][x_feat], errors="coerce"),
            y=pd.to_numeric(X.iloc[idx][y_feat], errors="coerce"),
            mode="markers",
            marker=dict(size=5),
            name="Observed batches",
        )
    )

    fig.update_layout(
        title=f"Contour explorer: {x_feat} vs {y_feat}",
        xaxis_title=x_feat,
        yaxis_title=y_feat,
        height=650,
    )
    st.plotly_chart(fig, width="stretch")
    st.caption("Other features held at median (numeric) or mode (categorical).")


# -----------------------------
# Single prediction
# -----------------------------
def single_input_form(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    X = df.drop(columns=[target_col])
    row: Dict[str, object] = {}

    st.write("Adjust process/material parameters and get an instant model prediction.")

    for col in X.columns:
        if X[col].dtype == "object" or str(X[col].dtype) == "category":
            opts = list(pd.Series(X[col]).dropna().unique())
            row[col] = st.selectbox(col, options=opts if opts else ["unknown"], index=0)
        else:
            s = pd.to_numeric(X[col], errors="coerce")
            p1 = float(np.nanpercentile(s, 1))
            p99 = float(np.nanpercentile(s, 99))
            med = float(np.nanmedian(s))
            if np.isfinite(p1) and np.isfinite(p99) and p1 != p99:
                row[col] = st.slider(col, min_value=p1, max_value=p99, value=med)
            else:
                row[col] = st.number_input(col, value=med)

    return pd.DataFrame([row])


# -----------------------------
# Main
# -----------------------------
def main():
    st.title("PharmaQualityAI: tablet quality prediction from manufacturing data")

    df = load_data()

    targets = detect_targets(df)
    cqa_present = [c for c in KNOWN_CQA_TARGETS if c in df.columns]

    with st.sidebar:
        st.header("Dataset")
        st.write(f"Loaded: {DATA_FILENAME}")
        st.write(f"Rows: {df.shape[0]}  Columns: {df.shape[1]}")

        st.header("Targets")
        if cqa_present:
            st.write("CQA targets detected:")
            st.write(cqa_present)
        else:
            st.write("No standard CQA names detected. Using numeric columns as candidates.")

        default_target = "Drug release average (%)" if "Drug release average (%)" in targets else targets[0]
        target = st.selectbox("Select target", options=targets, index=targets.index(default_target))

        st.header("Objective")
        objective = st.radio("Mode", options=["Regression", "Classification (out-of-spec)"])
        problem_mode = "regression" if objective == "Regression" else "classification"

        # Classification spec limits
        lower_spec = np.nan
        upper_spec = np.nan
        if problem_mode == "classification":
            st.subheader("Spec limits")
            st.write("Label: 1 = out-of-spec, 0 = within spec")
            colA, colB = st.columns(2)
            with colA:
                lower_spec = st.number_input("Lower spec (optional)", value=float("nan"))
            with colB:
                upper_spec = st.number_input("Upper spec (optional)", value=float("nan"))

        st.header("Model")
        if problem_mode == "regression":
            model_name = st.selectbox("Algorithm", options=["Random Forest", "ElasticNet"], index=0)
        else:
            model_name = st.selectbox("Algorithm", options=["Random Forest", "Logistic Regression"], index=0)

        test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
        seed = st.number_input("Random seed", value=42, step=1)

        scale_numeric = st.checkbox(
            "Scale numeric features",
            value=(model_name in {"ElasticNet", "Logistic Regression"}),
        )

    # Build working dataframe (classification adds a label column)
    work_df = df.copy()

    if problem_mode == "classification":
        y_num = pd.to_numeric(work_df[target], errors="coerce")

        # If no limits provided, auto-define abnormal batches using robust threshold
        no_limits = (not np.isfinite(lower_spec)) and (not np.isfinite(upper_spec))

        if no_limits:
            med = float(np.nanmedian(y_num))
            mad = float(np.nanmedian(np.abs(y_num - med)))
            thr = 3.0 * mad if mad > 0 else float(np.nanstd(y_num))
            if not np.isfinite(thr) or thr == 0:
                thr = float(np.nanstd(y_num)) if np.isfinite(np.nanstd(y_num)) else 1.0
            label = (np.abs(y_num - med) > thr).astype(int)
        else:
            label = pd.Series(0, index=work_df.index, dtype="int64")
            if np.isfinite(lower_spec):
                label = label | (y_num < lower_spec).astype(int)
            if np.isfinite(upper_spec):
                label = label | (y_num > upper_spec).astype(int)

        work_df["__out_of_spec__"] = label
        target_col = "__out_of_spec__"
    else:
        target_col = target

    tabs = st.tabs(["Data overview", "Model performance", "Single prediction", "Contour explorer"])

    with tabs[0]:
        st.subheader("Preview")
        st.dataframe(df.head(20), width="stretch")

        st.subheader("Missing values")
        mv = df.isna().sum().sort_values(ascending=False)
        mv = mv[mv > 0]
        if len(mv) == 0:
            st.write("No missing values detected.")
        else:
            st.dataframe(mv.to_frame("missing_count"), width="stretch")

        st.subheader("Distribution")
        num_cols = list(df.select_dtypes(include=[np.number]).columns)
        if num_cols:
            col = st.selectbox("Column", options=num_cols, index=num_cols.index(target) if target in num_cols else 0)
            fig = px.histogram(df, x=col, nbins=40, title=f"Distribution: {col}")
            st.plotly_chart(fig, width="stretch")

        st.subheader("Correlations")
        plot_correlation_heatmap(df)

    # Train pipeline
    pipe, X_train, X_test, y_train, y_test = train_pipeline(
        df=work_df,
        target_col=target_col,
        problem_mode=problem_mode,
        model_name=model_name,
        test_size=float(test_size),
        seed=int(seed),
        scale_numeric=scale_numeric,
    )

    with tabs[1]:
        st.subheader("Hold-out performance")
        plot_predictions(pipe, X_test, y_test, problem_mode)

        st.subheader("What drives predictions")
        plot_feature_importance(pipe, X_test, y_test, problem_mode)

    with tabs[2]:
        st.subheader("What-if prediction")
        if problem_mode == "regression":
            st.write(f"Target: {target}")
            input_df = single_input_form(work_df, target_col)
            pred = float(pipe.predict(input_df)[0])
            st.metric("Predicted value", f"{pred:.4f}")
        else:
            st.write(f"Risk label derived from: {target}")
            input_df = single_input_form(work_df, target_col)
            if hasattr(pipe, "predict_proba") and y_test.nunique() == 2:
                p = float(pipe.predict_proba(input_df)[0, 1])
                st.metric("Predicted p(out_of_spec)", f"{p:.4f}")
                st.metric("Predicted label (threshold 0.5)", str(int(p >= 0.5)))
            else:
                cls = int(pipe.predict(input_df)[0])
                st.metric("Predicted label", str(cls))

    with tabs[3]:
        st.subheader("Contour explorer")
        contour_explorer(pipe, work_df, target_col, problem_mode)


if __name__ == "__main__":
    main()
