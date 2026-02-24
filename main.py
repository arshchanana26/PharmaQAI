import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, roc_auc_score, classification_report, confusion_matrix
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.inspection import permutation_importance

import plotly.express as px
import plotly.graph_objects as go


st.set_page_config(page_title="Tablet Quality Prediction", layout="wide")

DEFAULT_PATH = "Process (1).csv"

KNOWN_TARGETS = [
    "Drug release average (%)",
    "Drug release min (%)",
    "Residual solvent",
    "Total impurities",
    "Impurity O",
    "Impurity L",
]
ID_COLS = ["batch", "code"]


def load_data(uploaded_file) -> pd.DataFrame:
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file, sep=";")
    return pd.read_csv(DEFAULT_PATH, sep=";")


def get_target_candidates(df: pd.DataFrame):
    # Prefer known CQA columns if present
    present_known = [c for c in KNOWN_TARGETS if c in df.columns]
    # Also include any columns with "impur", "release", "solvent" keywords
    kw = []
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in ["impur", "release", "solvent"]):
            if c not in present_known:
                kw.append(c)
    # Fallback: any numeric column not an ID
    numeric = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ID_COLS]
    # Build ordered unique list
    ordered = []
    for c in present_known + kw + numeric:
        if c not in ordered and c in df.columns:
            ordered.append(c)
    return ordered


def infer_feature_types(df: pd.DataFrame, target: str):
    X = df.drop(columns=[target])
    cat_cols = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype) == "category"]
    num_cols = [c for c in X.columns if c not in cat_cols]
    return num_cols, cat_cols


def build_preprocessor(num_cols, cat_cols, scale_numeric: bool):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(steps=numeric_steps)

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )


def get_model(problem_mode: str, model_name: str, random_state: int):
    if problem_mode == "regression":
        if model_name == "Random Forest (regression)":
            return RandomForestRegressor(
                n_estimators=400,
                random_state=random_state,
                n_jobs=-1
            )
        if model_name == "ElasticNet (regression)":
            return ElasticNet(random_state=random_state)
        raise ValueError("Unknown regression model")
    else:
        if model_name == "Random Forest (classification)":
            return RandomForestClassifier(
                n_estimators=400,
                random_state=random_state,
                n_jobs=-1,
                class_weight="balanced"
            )
        if model_name == "Logistic Regression (classification)":
            return LogisticRegression(
                max_iter=5000,
                class_weight="balanced"
            )
        raise ValueError("Unknown classification model")


def regression_metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse
    }


def classification_metrics(y_true, y_pred, y_proba=None):
    out = {"Accuracy": float(accuracy_score(y_true, y_pred))}
    if y_proba is not None and len(np.unique(y_true)) == 2:
        out["ROC AUC"] = float(roc_auc_score(y_true, y_proba))
    return out


def make_correlation_heatmap(df: pd.DataFrame):
    num = df.select_dtypes(include=[np.number]).copy()
    if num.shape[1] < 2:
        st.info("Not enough numeric columns for correlation heatmap.")
        return
    corr = num.corr(numeric_only=True)
    fig = px.imshow(corr, aspect="auto", title="Correlation heatmap (numeric columns)")
    st.plotly_chart(fig, use_container_width=True)


def get_feature_names(preprocessor: ColumnTransformer):
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return []


@st.cache_data(show_spinner=False)
def prepare_xy(df: pd.DataFrame, target: str):
    data = df.dropna(subset=[target]).copy()
    X = data.drop(columns=[target])
    y = data[target].copy()
    return X, y


@st.cache_resource(show_spinner=False)
def train_pipeline(df: pd.DataFrame, target_col: str, problem_mode: str,
                   test_size: float, random_state: int,
                   model_name: str, scale_numeric: bool):
    X, y = prepare_xy(df, target_col)

    # stratify only for classification, if few classes
    stratify = y if (problem_mode == "classification" and y.nunique() <= 20) else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    num_cols, cat_cols = infer_feature_types(pd.concat([X, y], axis=1), target_col)
    pre = build_preprocessor(num_cols, cat_cols, scale_numeric=scale_numeric)
    model = get_model(problem_mode, model_name=model_name, random_state=random_state)

    pipe = Pipeline(steps=[("preprocessor", pre), ("model", model)])
    pipe.fit(X_train, y_train)
    return pipe, X_train, X_test, y_train, y_test


def permutation_importance_plot(pipe: Pipeline, X_test, y_test, problem_mode: str):
    pre = pipe.named_steps["preprocessor"]
    feature_names = get_feature_names(pre)
    if not feature_names:
        st.info("Feature names could not be extracted for importance.")
        return

    if problem_mode == "regression":
        scoring = "r2"
    else:
        scoring = "roc_auc" if y_test.nunique() == 2 else "accuracy"

    try:
        r = permutation_importance(
            pipe, X_test, y_test,
            n_repeats=15, random_state=0, n_jobs=-1, scoring=scoring
        )
    except Exception as e:
        st.warning(f"Could not compute permutation importance: {e}")
        return

    imp = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": r.importances_mean,
        "importance_std": r.importances_std
    }).sort_values("importance_mean", ascending=False).head(25)

    fig = px.bar(
        imp[::-1],
        x="importance_mean",
        y="feature",
        orientation="h",
        error_x="importance_std",
        title="Permutation importance (top 25)"
    )
    st.plotly_chart(fig, use_container_width=True)


def prediction_plots(pipe: Pipeline, X_test, y_test, problem_mode: str):
    if problem_mode == "regression":
        y_pred = pipe.predict(X_test)
        met = regression_metrics(y_test, y_pred)

        c1, c2, c3 = st.columns(3)
        c1.metric("R2", f"{met['R2']:.4f}")
        c2.metric("MAE", f"{met['MAE']:.4f}")
        c3.metric("RMSE", f"{met['RMSE']:.4f}")

        dfp = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
        fig1 = px.scatter(dfp, x="Actual", y="Predicted", trendline="ols", title="Actual vs Predicted")
        st.plotly_chart(fig1, use_container_width=True)

        resid = y_test.values - y_pred
        fig2 = px.scatter(
            x=y_pred, y=resid,
            labels={"x": "Predicted", "y": "Residual"},
            title="Residuals vs Predicted"
        )
        st.plotly_chart(fig2, use_container_width=True)

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
        st.plotly_chart(fig, use_container_width=True)

        if y_proba is not None:
            fig2 = px.histogram(
                pd.DataFrame({"p(out_of_spec)": y_proba}),
                x="p(out_of_spec)",
                nbins=30,
                title="Predicted probability distribution"
            )
            st.plotly_chart(fig2, use_container_width=True)


def make_single_prediction_input(df: pd.DataFrame, target: str):
    X = df.drop(columns=[target])
    row = {}
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


def contour_explorer(pipe: Pipeline, df: pd.DataFrame, target: str, problem_mode: str):
    data = df.dropna(subset=[target]).copy()
    X = data.drop(columns=[target])
    y = data[target]

    num_cols = list(X.select_dtypes(include=[np.number]).columns)
    if len(num_cols) < 2:
        st.info("Need at least two numeric features for contour plots.")
        return

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        x_feat = st.selectbox("X feature", options=num_cols, index=0)
    with c2:
        y_feat = st.selectbox("Y feature", options=[c for c in num_cols if c != x_feat], index=0)
    with c3:
        grid_n = st.slider("Grid resolution", min_value=25, max_value=120, value=60, step=5)

    fixed = {}
    for col in X.columns:
        if col in [x_feat, y_feat]:
            continue
        if X[col].dtype == "object" or str(X[col].dtype) == "category":
            fixed[col] = pd.Series(X[col]).dropna().mode().iloc[0] if X[col].dropna().shape[0] else "unknown"
        else:
            fixed[col] = float(pd.to_numeric(X[col], errors="coerce").median())

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
        z_title = f"Predicted {target}"
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
    fig.add_trace(go.Scatter(
        x=pd.to_numeric(X.iloc[idx][x_feat], errors="coerce"),
        y=pd.to_numeric(X.iloc[idx][y_feat], errors="coerce"),
        mode="markers",
        marker=dict(size=5),
        name="Observed batches"
    ))

    fig.update_layout(
        title=f"Contour explorer: {x_feat} vs {y_feat}",
        xaxis_title=x_feat,
        yaxis_title=y_feat,
        height=650
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Contour computed with other features held at median (numeric) or mode (categorical).")


def main():
    st.title("Machine learning prediction of tablet quality attributes")

    with st.sidebar:
        st.header("Data")
        uploaded = st.file_uploader("Upload CSV (semicolon-separated)", type=["csv"])
        df = load_data(uploaded)

        st.header("Targets detected in your file")
        targets = get_target_candidates(df)

        # Show the CQA targets clearly
        cqa_present = [c for c in KNOWN_TARGETS if c in df.columns]
        st.write("CQA columns found:")
        st.write(cqa_present if cqa_present else "None found (unexpected for your file).")

        # pick default target: Drug release average (%), else first CQA, else first candidate
        default_target = "Drug release average (%)" if "Drug release average (%)" in targets else (cqa_present[0] if cqa_present else targets[0])
        target = st.selectbox("Choose target (what you want to predict)", options=targets, index=targets.index(default_target))

        st.header("Objective")
        objective = st.radio("Type", options=["Regression", "Classification (out-of-spec)"])
        problem_mode = "regression" if objective == "Regression" else "classification"

        lower_spec, upper_spec = None, None
        if problem_mode == "classification":
            st.subheader("Specification limits")
            st.write("Label: 1 = out-of-spec, 0 = within-spec")
            colA, colB = st.columns(2)
            with colA:
                lower_spec = st.number_input("Lower spec (optional)", value=float("nan"))
            with colB:
                upper_spec = st.number_input("Upper spec (optional)", value=float("nan"))

        st.header("Training")
        model_name = st.selectbox(
            "Model",
            options=(["Random Forest (regression)", "ElasticNet (regression)"]
                     if problem_mode == "regression"
                     else ["Random Forest (classification)", "Logistic Regression (classification)"])
        )
        test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
        random_state = st.number_input("Random seed", value=42, step=1)
        scale_numeric = st.checkbox(
            "Scale numeric features (useful for linear models)",
            value=(model_name.startswith("ElasticNet") or model_name.startswith("Logistic"))
        )

    # Prepare working dataframe
    work_df = df.copy()

    # For classification: create binary out-of-spec label from chosen numeric target
    if problem_mode == "classification":
        y_num = pd.to_numeric(work_df[target], errors="coerce")
        label = pd.Series(0, index=work_df.index, dtype="int64")
        if lower_spec is not None and np.isfinite(lower_spec):
            label = label | (y_num < lower_spec).astype(int)
        if upper_spec is not None and np.isfinite(upper_spec):
            label = label | (y_num > upper_spec).astype(int)

        # If user didn't enter any specs, fallback to a robust "abnormal" definition
        if (lower_spec is None or not np.isfinite(lower_spec)) and (upper_spec is None or not np.isfinite(upper_spec)):
            med = float(np.nanmedian(y_num))
            mad = float(np.nanmedian(np.abs(y_num - med)))
            thr = 3.0 * mad if mad > 0 else float(np.nanstd(y_num))
            if not np.isfinite(thr) or thr == 0:
                thr = float(np.nanstd(y_num)) if np.isfinite(np.nanstd(y_num)) else 1.0
            label = (np.abs(y_num - med) > thr).astype(int)

        work_df["__out_of_spec__"] = label
        target_col = "__out_of_spec__"
    else:
        target_col = target

    tabs = st.tabs(["Data overview", "Model performance", "Single prediction", "Contour explorer"])

    with tabs[0]:
        st.subheader("Preview")
        st.dataframe(df.head(20), use_container_width=True)

        st.subheader("Missing values")
        mv = df.isna().sum().sort_values(ascending=False)
        st.dataframe(mv[mv > 0].to_frame("missing_count"), use_container_width=True)

        st.subheader("Interactive distribution")
        numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
        col = st.selectbox("Column for distribution plot", options=numeric_cols, index=numeric_cols.index(target) if target in numeric_cols else 0)
        fig = px.histogram(df, x=col, nbins=40, title=f"Distribution: {col}")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Correlations")
        make_correlation_heatmap(df)

    # Train model
    pipe, X_train, X_test, y_train, y_test = train_pipeline(
        work_df.drop(columns=[target]) if (problem_mode == "classification") else work_df,
        target_col=target_col,
        problem_mode=problem_mode,
        test_size=float(test_size),
        random_state=int(random_state),
        model_name=model_name,
        scale_numeric=scale_numeric
    )

    with tabs[1]:
        st.subheader("Performance on hold-out test set")
        prediction_plots(pipe, X_test, y_test, problem_mode)

        st.subheader("Feature importance")
        permutation_importance_plot(pipe, X_test, y_test, problem_mode)

    with tabs[2]:
        st.subheader("What-if prediction")
        if problem_mode == "regression":
            st.write(f"Predicting: {target}")
        else:
            st.write(f"Predicting out-of-spec risk label derived from: {target}")

        base_df = work_df.copy()
        if problem_mode == "classification":
            base_df = base_df.drop(columns=["__out_of_spec__"])

        input_df = make_single_prediction_input(base_df, target if problem_mode == "regression" else target)

        if problem_mode == "regression":
            pred = float(pipe.predict(input_df)[0])
            st.metric("Predicted value", f"{pred:.4f}")
        else:
            if hasattr(pipe, "predict_proba"):
                p = float(pipe.predict_proba(input_df)[0, 1])
                st.metric("Predicted p(out_of_spec)", f"{p:.4f}")
                st.metric("Predicted label (threshold 0.5)", str(int(p >= 0.5)))
            else:
                cls = int(pipe.predict(input_df)[0])
                st.metric("Predicted label", str(cls))

    with tabs[3]:
        st.subheader("Contour-based process exploration")
        contour_explorer(
            pipe=pipe,
            df=work_df.drop(columns=[target]) if problem_mode == "classification" else work_df,
            target=target_col,
            problem_mode=problem_mode
        )


if __name__ == "__main__":
    main()
