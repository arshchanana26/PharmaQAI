import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)

# Regression models
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import io

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PharmaQAI — Process Intelligence",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLE
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = {
    "bg":        "#0d1117",
    "panel":     "#161b22",
    "border":    "#30363d",
    "accent1":   "#58a6ff",
    "accent2":   "#3fb950",
    "accent3":   "#f78166",
    "accent4":   "#d2a8ff",
    "accent5":   "#ffa657",
    "text":      "#e6edf3",
    "muted":     "#8b949e",
}

CMAP_CONTOUR  = "plasma"
CMAP_HEATMAP  = "viridis"
CMAP_SCATTER  = "coolwarm"

MPL_STYLE = {
    "figure.facecolor":     PALETTE["bg"],
    "axes.facecolor":       PALETTE["panel"],
    "axes.edgecolor":       PALETTE["border"],
    "axes.labelcolor":      PALETTE["text"],
    "axes.titlecolor":      PALETTE["text"],
    "xtick.color":          PALETTE["muted"],
    "ytick.color":          PALETTE["muted"],
    "text.color":           PALETTE["text"],
    "grid.color":           PALETTE["border"],
    "grid.linestyle":       "--",
    "grid.alpha":           0.6,
    "legend.facecolor":     PALETTE["panel"],
    "legend.edgecolor":     PALETTE["border"],
    "legend.labelcolor":    PALETTE["text"],
    "figure.dpi":           130,
}
plt.rcParams.update(MPL_STYLE)

LABEL_STYLE = dict(fontsize=9, fontweight="normal", color=PALETTE["text"])
TITLE_STYLE = dict(fontsize=11, fontweight="normal", color=PALETTE["text"], pad=10)

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

  html, body, [class*="css"] {{
      font-family: 'IBM Plex Sans', sans-serif;
      background-color: {PALETTE['bg']};
      color: {PALETTE['text']};
  }}
  .block-container {{ padding: 1.5rem 2rem 3rem 2rem; max-width: 1400px; }}
  section[data-testid="stSidebar"] {{
      background-color: {PALETTE['panel']};
      border-right: 1px solid {PALETTE['border']};
  }}
  /* Force ALL text white/light everywhere */
  *, *::before, *::after {{
      color: {PALETTE['text']};
  }}
  h1, h2, h3 {{
      font-family: 'IBM Plex Mono', monospace;
      letter-spacing: -0.03em;
      color: {PALETTE['text']} !important;
  }}

  /* Sidebar — every element forced light */
  section[data-testid="stSidebar"] * {{
      color: {PALETTE['text']} !important;
  }}
  section[data-testid="stSidebar"] p,
  section[data-testid="stSidebar"] span,
  section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] div {{
      color: {PALETTE['text']} !important;
  }}
  /* Radio labels */
  div[role="radiogroup"] label p,
  div[role="radiogroup"] label span {{
      color: {PALETTE['text']} !important;
  }}
  /* File uploader inner text */
  .stFileUploader span, .stFileUploader p, .stFileUploader label {{
      color: {PALETTE['text']} !important;
  }}

  /* Widget labels everywhere */
  label, .stSelectbox label p, .stMultiSelect label p,
  .stSlider label p, .stTextInput label p {{
      color: {PALETTE['text']} !important;
  }}
  /* Selectbox selected value text */
  div[data-baseweb="select"] span {{
      color: {PALETTE['text']} !important;
  }}

  .metric-card {{
      background: {PALETTE['panel']};
      border: 1px solid {PALETTE['border']};
      border-radius: 8px;
      padding: 1rem 1.25rem;
      margin-bottom: 0.5rem;
  }}
  .metric-value {{
      font-family: 'IBM Plex Mono', monospace;
      font-size: 1.6rem;
      color: {PALETTE['accent1']} !important;
  }}
  .metric-label {{
      font-size: 0.75rem;
      color: {PALETTE['muted']} !important;
      text-transform: uppercase;
      letter-spacing: 0.08em;
  }}
  .tag {{
      display: inline-block;
      background: {PALETTE['border']};
      color: {PALETTE['accent1']} !important;
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.7rem;
      padding: 2px 8px;
      border-radius: 12px;
      margin: 2px;
  }}
  hr {{ border-color: {PALETTE['border']}; }}
  .stButton > button {{
      background: {PALETTE['accent1']};
      color: {PALETTE['bg']} !important;
      border: none;
      border-radius: 6px;
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.8rem;
      font-weight: 600;
      padding: 0.45rem 1.2rem;
      transition: opacity 0.15s;
  }}
  .stButton > button:hover {{ opacity: 0.85; }}
  .stAlert p {{ color: {PALETTE['text']} !important; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def fig_to_st(fig, key=None):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    st.image(buf, use_container_width=True)
    plt.close(fig)

def metric_card(label, value, color=None):
    c = color or PALETTE["accent1"]
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value" style="color:{c}">{value}</div>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    df["weekend"] = df["weekend"].map({"yes": 1, "no": 0}).astype(int)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## PharmaQAI")
    st.markdown("<span class='tag'>Process Intelligence</span> <span class='tag'>ML</span>", unsafe_allow_html=True)
    st.markdown("---")

    uploaded = st.file_uploader("Upload dataset (CSV, semicolon-delimited)", type=["csv"])
    if uploaded:
        df_raw = pd.read_csv(uploaded, sep=";")
        df_raw["weekend"] = df_raw["weekend"].map({"yes": 1, "no": 0}).astype(int)
    else:
        df_raw = load_data("Process (1).csv")

    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["Data Overview", "Regression", "Classification", "Contour Explorer"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    muted = PALETTE["muted"]
    st.markdown(f"<span style='color:{muted};font-size:0.72rem'>1005 batches - 35 features<br>Pharmaceutical tablet manufacturing</span>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE / TARGET DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
PROCESS_FEATURES = [
    "tbl_speed_mean", "tbl_speed_change", "tbl_speed_0_duration",
    "total_waste", "startup_waste", "fom_mean", "fom_change",
    "SREL_startup_mean", "SREL_production_mean", "SREL_production_max",
    "main_CompForce mean", "main_CompForce_sd", "main_CompForce_median",
    "pre_CompForce_mean", "tbl_fill_mean", "tbl_fill_sd",
    "cyl_height_mean", "stiffness_mean", "stiffness_max", "stiffness_min",
    "ejection_mean", "ejection_max", "ejection_min",
    "Startup_tbl_fill_maxDifference", "Startup_main_CompForce_mean",
    "Startup_tbl_fill_mean", "weekend",
]

QUALITY_TARGETS = [
    "Drug release average (%)", "Drug release min (%)",
    "Residual solvent", "Total impurities", "Impurity O", "Impurity L",
]

REGRESSION_MODELS = {
    "Ridge Regression":        Ridge(alpha=1.0),
    "Lasso Regression":        Lasso(alpha=0.01, max_iter=5000),
    "Elastic Net":             ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000),
    "Random Forest":           RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting":       GradientBoostingRegressor(n_estimators=200, random_state=42),
    "Extra Trees":             ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "SVR (RBF)":               SVR(kernel="rbf", C=10, gamma="scale"),
    "KNN Regressor":           KNeighborsRegressor(n_neighbors=7),
    "PLS Regression":          PLSRegression(n_components=5),
}

CLASSIFICATION_MODELS = {
    "Logistic Regression":     LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":           RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting":       GradientBoostingClassifier(n_estimators=200, random_state=42),
    "SVM (RBF)":               SVC(kernel="rbf", probability=True, random_state=42),
    "KNN Classifier":          KNeighborsClassifier(n_neighbors=7),
    "LDA":                     LinearDiscriminantAnalysis(),
}

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — DATA OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if page == "Data Overview":
    st.markdown("# Data Overview")
    st.markdown(f"<span style='color:{PALETTE['muted']}'>Pharmaceutical tablet manufacturing — process & quality parameters</span>", unsafe_allow_html=True)
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Batches", f"{len(df_raw):,}")
    with c2: metric_card("Features", str(len(PROCESS_FEATURES)), PALETTE["accent2"])
    with c3: metric_card("Quality targets", str(len(QUALITY_TARGETS)), PALETTE["accent4"])
    with c4: metric_card("Product codes", str(df_raw["code"].nunique()), PALETTE["accent3"])

    st.markdown("---")
    st.markdown("### Sample data")
    st.dataframe(df_raw.head(20), use_container_width=True, height=280)

    st.markdown("### Descriptive statistics")
    st.dataframe(df_raw[PROCESS_FEATURES + QUALITY_TARGETS].describe().round(3), use_container_width=True, height=280)

    st.markdown("---")
    st.markdown("### Quality target distributions")

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    colors = [PALETTE["accent1"], PALETTE["accent2"], PALETTE["accent3"],
              PALETTE["accent4"], PALETTE["accent5"], "#63e6be"]
    for ax, col, c in zip(axes.flat, QUALITY_TARGETS, colors):
        data = df_raw[col].dropna()
        ax.hist(data, bins=35, color=c, alpha=0.85, edgecolor=PALETTE["bg"], linewidth=0.4)
        ax.axvline(data.mean(), color="white", linewidth=1.2, linestyle="--", alpha=0.7)
        ax.set_title(col, **TITLE_STYLE)
        ax.set_xlabel("Value", **LABEL_STYLE)
        ax.set_ylabel("Count", **LABEL_STYLE)
        ax.grid(True, axis="y")
    plt.tight_layout(pad=1.5)
    fig_to_st(fig)

    st.markdown("### Correlation heatmap — quality targets")
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df_raw[QUALITY_TARGETS].corr()
    n = len(QUALITY_TARGETS)
    im = ax.imshow(corr.values, cmap=CMAP_HEATMAP, vmin=-1, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    short = [c.replace("Drug release ", "DR ").replace(" (%)", "").replace("Total ", "Tot. ") for c in QUALITY_TARGETS]
    ax.set_xticklabels(short, rotation=35, ha="right", **LABEL_STYLE)
    ax.set_yticklabels(short, **LABEL_STYLE)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if abs(corr.values[i,j]) > 0.5 else PALETTE["muted"])
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    ax.set_title("Quality Target Correlations", **TITLE_STYLE)
    plt.tight_layout()
    fig_to_st(fig)

    st.markdown("### Feature correlation with quality targets (top 10 per target)")
    sel_qt = st.selectbox("Select quality target", QUALITY_TARGETS)
    num_feats = [f for f in PROCESS_FEATURES if df_raw[f].dtype != object]
    corrs = df_raw[num_feats].corrwith(df_raw[sel_qt]).dropna().abs().sort_values(ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(corrs.index[::-1], corrs.values[::-1],
                   color=[PALETTE["accent1"] if v > 0.3 else PALETTE["muted"] for v in corrs.values[::-1]],
                   edgecolor=PALETTE["bg"], linewidth=0.4)
    for bar, val in zip(bars, corrs.values[::-1]):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8, color=PALETTE["text"])
    ax.set_xlabel("|Pearson r|", **LABEL_STYLE)
    ax.set_title(f"Top correlations with {sel_qt}", **TITLE_STYLE)
    ax.set_xlim(0, corrs.max() + 0.1)
    ax.grid(True, axis="x")
    plt.tight_layout()
    fig_to_st(fig)

    st.markdown("### Batch distribution by product code")
    code_counts = df_raw["code"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    cmap_vals = cm.plasma(np.linspace(0.2, 0.9, len(code_counts)))
    ax.bar(code_counts.index.astype(str), code_counts.values, color=cmap_vals,
           edgecolor=PALETTE["bg"], linewidth=0.5)
    ax.set_xlabel("Product code", **LABEL_STYLE)
    ax.set_ylabel("Number of batches", **LABEL_STYLE)
    ax.set_title("Batch count per product code", **TITLE_STYLE)
    ax.grid(True, axis="y")
    plt.tight_layout()
    fig_to_st(fig)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — REGRESSION
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Regression":
    st.markdown("# Regression — Quality Prediction")
    st.markdown(f"<span style='color:{PALETTE['muted']}'>Predict continuous quality attributes from process parameters</span>", unsafe_allow_html=True)
    st.markdown("---")

    col_s, col_m = st.columns([1, 1])
    with col_s:
        target = st.selectbox("Target variable", QUALITY_TARGETS)
    with col_m:
        model_name = st.selectbox("Algorithm", list(REGRESSION_MODELS.keys()))

    feat_options = [f for f in PROCESS_FEATURES if df_raw[f].dtype != object]
    sel_feats = st.multiselect("Input features", feat_options, default=feat_options[:12])

    c1, c2, c3 = st.columns(3)
    test_size  = c1.slider("Test set size", 0.10, 0.40, 0.20, 0.05)
    cv_folds   = c2.slider("CV folds", 3, 10, 5)
    run_btn    = c3.button("Run regression", use_container_width=True)

    if run_btn:
        if len(sel_feats) < 2:
            st.warning("Select at least 2 features.")
            st.stop()

        df_model = df_raw[sel_feats + [target]].dropna()
        X = df_model[sel_feats].values
        y = df_model[target].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42)

        model = REGRESSION_MODELS[model_name]
        if isinstance(model, PLSRegression):
            model.set_params(n_components=min(5, len(sel_feats)))

        pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        if y_pred.ndim > 1:
            y_pred = y_pred.ravel()

        r2   = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae  = mean_absolute_error(y_test, y_pred)
        cv_r2 = cross_val_score(pipe, X, y, cv=cv_folds, scoring="r2").mean()

        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        with m1: metric_card("R²", f"{r2:.4f}", PALETTE["accent2"] if r2 > 0.7 else PALETTE["accent3"])
        with m2: metric_card("RMSE", f"{rmse:.4f}", PALETTE["accent1"])
        with m3: metric_card("MAE",  f"{mae:.4f}",  PALETTE["accent4"])
        with m4: metric_card(f"CV R² ({cv_folds}-fold)", f"{cv_r2:.4f}", PALETTE["accent5"])

        st.markdown("---")
        st.markdown("### Predicted vs Actual")

        residuals = y_test - y_pred
        fig = plt.figure(figsize=(14, 5))
        gs  = GridSpec(1, 3, figure=fig, wspace=0.35)

        # Scatter predicted vs actual
        ax1 = fig.add_subplot(gs[0])
        sc = ax1.scatter(y_test, y_pred, c=residuals, cmap=CMAP_SCATTER,
                         s=30, alpha=0.75, edgecolors="none")
        lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        ax1.plot([lo, hi], [lo, hi], color=PALETTE["accent2"], linewidth=1.5, linestyle="--", label="Perfect fit")
        plt.colorbar(sc, ax=ax1, fraction=0.04, pad=0.04, label="Residual")
        ax1.set_xlabel("Actual", **LABEL_STYLE)
        ax1.set_ylabel("Predicted", **LABEL_STYLE)
        ax1.set_title(f"{model_name}\nPredicted vs Actual", **TITLE_STYLE)
        ax1.legend(fontsize=8)
        ax1.grid(True)

        # Residual distribution
        ax2 = fig.add_subplot(gs[1])
        ax2.hist(residuals, bins=30, color=PALETTE["accent1"], edgecolor=PALETTE["bg"],
                 linewidth=0.4, alpha=0.9)
        ax2.axvline(0, color=PALETTE["accent3"], linewidth=1.5, linestyle="--")
        ax2.set_xlabel("Residual", **LABEL_STYLE)
        ax2.set_ylabel("Count", **LABEL_STYLE)
        ax2.set_title("Residual Distribution", **TITLE_STYLE)
        ax2.grid(True, axis="y")

        # Residuals vs predicted
        ax3 = fig.add_subplot(gs[2])
        ax3.scatter(y_pred, residuals, c=PALETTE["accent4"], s=28, alpha=0.7, edgecolors="none")
        ax3.axhline(0, color=PALETTE["accent3"], linewidth=1.5, linestyle="--")
        ax3.set_xlabel("Predicted", **LABEL_STYLE)
        ax3.set_ylabel("Residual", **LABEL_STYLE)
        ax3.set_title("Residuals vs Fitted", **TITLE_STYLE)
        ax3.grid(True)

        fig_to_st(fig)

        # Feature importance (if available)
        has_fi = hasattr(pipe.named_steps["model"], "feature_importances_")
        has_coef = hasattr(pipe.named_steps["model"], "coef_")
        if has_fi or has_coef:
            st.markdown("### Feature importance")
            if has_fi:
                importances = pipe.named_steps["model"].feature_importances_
            else:
                coef = pipe.named_steps["model"].coef_
                if coef.ndim > 1:
                    coef = coef[0]
                importances = np.abs(coef)

            idx = np.argsort(importances)[::-1][:15]
            fig, ax = plt.subplots(figsize=(10, 4))
            bar_c = cm.plasma(np.linspace(0.2, 0.85, len(idx)))
            ax.bar(range(len(idx)), importances[idx], color=bar_c, edgecolor=PALETTE["bg"], linewidth=0.4)
            ax.set_xticks(range(len(idx)))
            ax.set_xticklabels([sel_feats[i] for i in idx], rotation=40, ha="right", fontsize=8, color=PALETTE["text"])
            ax.set_ylabel("Importance", **LABEL_STYLE)
            ax.set_title(f"Top feature importances — {model_name}", **TITLE_STYLE)
            ax.grid(True, axis="y")
            plt.tight_layout()
            fig_to_st(fig)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Classification":
    st.markdown("# Classification — Batch Categorisation")
    st.markdown(f"<span style='color:{PALETTE['muted']}'>Predict batch categories from process parameters</span>", unsafe_allow_html=True)
    st.markdown("---")

    clf_targets = {
        "Weekend batch (yes/no)": "weekend",
        "Product code (multi-class)": "code",
    }
    c1, c2 = st.columns(2)
    clf_target_label = c1.selectbox("Classification target", list(clf_targets.keys()))
    clf_model_name   = c2.selectbox("Algorithm", list(CLASSIFICATION_MODELS.keys()))

    feat_options = [f for f in PROCESS_FEATURES if df_raw[f].dtype != object and f != clf_targets[clf_target_label]]
    sel_feats = st.multiselect("Input features", feat_options, default=feat_options[:12])

    c3, c4, c5 = st.columns(3)
    test_size = c3.slider("Test set size", 0.10, 0.40, 0.20, 0.05)
    cv_folds  = c4.slider("CV folds", 3, 10, 5)
    run_btn   = c5.button("Run classification", use_container_width=True)

    if run_btn:
        if len(sel_feats) < 2:
            st.warning("Select at least 2 features.")
            st.stop()

        clf_col = clf_targets[clf_target_label]
        df_model = df_raw[sel_feats + [clf_col]].dropna()
        X = df_model[sel_feats].values
        y = df_model[clf_col].values
        classes = np.unique(y)
        is_binary = len(classes) == 2

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y)

        model = CLASSIFICATION_MODELS[clf_model_name]
        pipe  = Pipeline([("scaler", StandardScaler()), ("model", model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        acc  = accuracy_score(y_test, y_pred)
        cv_acc = cross_val_score(pipe, X, y, cv=cv_folds, scoring="accuracy").mean()

        if is_binary and hasattr(pipe, "predict_proba"):
            y_prob = pipe.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
        else:
            auc = None

        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        with m1: metric_card("Accuracy", f"{acc:.4f}", PALETTE["accent2"] if acc > 0.8 else PALETTE["accent3"])
        with m2: metric_card(f"CV Acc ({cv_folds}-fold)", f"{cv_acc:.4f}", PALETTE["accent1"])
        with m3: metric_card("Classes", str(len(classes)), PALETTE["accent4"])
        with m4: metric_card("AUC-ROC", f"{auc:.4f}" if auc else "N/A", PALETTE["accent5"])

        st.markdown("---")
        fig = plt.figure(figsize=(14, 5 if not is_binary else 5))
        gs  = GridSpec(1, 3 if is_binary else 2, figure=fig, wspace=0.38)

        # Confusion matrix
        ax1 = fig.add_subplot(gs[0])
        cm_mat = confusion_matrix(y_test, y_pred, labels=classes)
        cm_norm = cm_mat.astype(float) / cm_mat.sum(axis=1, keepdims=True)
        im = ax1.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax1.set_xticks(range(len(classes))); ax1.set_yticks(range(len(classes)))
        clabels = [str(c) for c in classes]
        ax1.set_xticklabels(clabels, rotation=45, ha="right", fontsize=8, color=PALETTE["text"])
        ax1.set_yticklabels(clabels, fontsize=8, color=PALETTE["text"])
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax1.text(j, i, f"{cm_mat[i,j]}", ha="center", va="center",
                         fontsize=8, color="white" if cm_norm[i,j] > 0.5 else PALETTE["muted"])
        plt.colorbar(im, ax=ax1, fraction=0.04, pad=0.04)
        ax1.set_xlabel("Predicted", **LABEL_STYLE)
        ax1.set_ylabel("Actual", **LABEL_STYLE)
        ax1.set_title("Confusion Matrix (normalised)", **TITLE_STYLE)

        # Per-class accuracy bar
        ax2 = fig.add_subplot(gs[1])
        per_class_acc = cm_norm.diagonal()
        bar_c = cm.viridis(np.linspace(0.3, 0.9, len(classes)))
        ax2.bar(clabels, per_class_acc, color=bar_c, edgecolor=PALETTE["bg"], linewidth=0.4)
        ax2.set_ylim(0, 1.05)
        ax2.axhline(acc, color=PALETTE["accent3"], linestyle="--", linewidth=1.2, label=f"Overall acc={acc:.3f}")
        ax2.set_xlabel("Class", **LABEL_STYLE)
        ax2.set_ylabel("Recall", **LABEL_STYLE)
        ax2.set_title("Per-class recall", **TITLE_STYLE)
        ax2.legend(fontsize=8)
        ax2.grid(True, axis="y")

        # ROC curve for binary
        if is_binary:
            ax3 = fig.add_subplot(gs[2])
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            ax3.plot(fpr, tpr, color=PALETTE["accent1"], linewidth=2.0, label=f"AUC = {auc:.3f}")
            ax3.plot([0, 1], [0, 1], color=PALETTE["muted"], linestyle="--", linewidth=1)
            ax3.fill_between(fpr, tpr, alpha=0.15, color=PALETTE["accent1"])
            ax3.set_xlabel("False positive rate", **LABEL_STYLE)
            ax3.set_ylabel("True positive rate", **LABEL_STYLE)
            ax3.set_title("ROC Curve", **TITLE_STYLE)
            ax3.legend(fontsize=9)
            ax3.grid(True)

        fig_to_st(fig)

        # Feature importance
        has_fi   = hasattr(pipe.named_steps["model"], "feature_importances_")
        has_coef = hasattr(pipe.named_steps["model"], "coef_")
        if has_fi or has_coef:
            st.markdown("### Feature importance")
            if has_fi:
                importances = pipe.named_steps["model"].feature_importances_
            else:
                coef = pipe.named_steps["model"].coef_
                if coef.ndim > 1:
                    coef = np.abs(coef).mean(axis=0)
                importances = np.abs(coef)
            idx = np.argsort(importances)[::-1][:15]
            fig, ax = plt.subplots(figsize=(10, 4))
            bar_c = cm.plasma(np.linspace(0.15, 0.85, len(idx)))
            ax.bar(range(len(idx)), importances[idx], color=bar_c, edgecolor=PALETTE["bg"], linewidth=0.4)
            ax.set_xticks(range(len(idx)))
            ax.set_xticklabels([sel_feats[i] for i in idx], rotation=40, ha="right", fontsize=8, color=PALETTE["text"])
            ax.set_ylabel("Importance", **LABEL_STYLE)
            ax.set_title(f"Top feature importances — {clf_model_name}", **TITLE_STYLE)
            ax.grid(True, axis="y")
            plt.tight_layout()
            fig_to_st(fig)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — CONTOUR EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Contour Explorer":
    st.markdown("# Contour Explorer")
    st.markdown(f"<span style='color:{PALETTE['muted']}'>Visualise how two process parameters jointly influence a quality attribute using a trained ML model</span>", unsafe_allow_html=True)
    st.markdown("---")

    feat_options = [f for f in PROCESS_FEATURES if df_raw[f].dtype != object]

    c1, c2, c3 = st.columns(3)
    x_feat  = c1.selectbox("X-axis feature",   feat_options, index=0)
    y_feat  = c2.selectbox("Y-axis feature",    feat_options, index=3)
    c_target = c3.selectbox("Response (target)", QUALITY_TARGETS, index=0)

    c4, c5, c6 = st.columns(3)
    model_name = c4.selectbox("Model", list(REGRESSION_MODELS.keys()), index=3)
    grid_res   = c5.slider("Grid resolution", 30, 120, 60, 10)
    run_contour = c6.button("Generate contour", use_container_width=True)

    if run_contour:
        if x_feat == y_feat:
            st.warning("X and Y features must be different.")
            st.stop()

        remaining_feats = [f for f in feat_options if f not in (x_feat, y_feat)]
        df_model = df_raw[feat_options + [c_target]].dropna()
        X_all = df_model[feat_options].values
        y_all = df_model[c_target].values

        model = REGRESSION_MODELS[model_name]
        if isinstance(model, PLSRegression):
            model.set_params(n_components=min(5, len(feat_options)))

        pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
        pipe.fit(X_all, y_all)
        y_pred_train = pipe.predict(X_all)
        if y_pred_train.ndim > 1:
            y_pred_train = y_pred_train.ravel()
        train_r2 = r2_score(y_all, y_pred_train)

        xi = feat_options.index(x_feat)
        yi = feat_options.index(y_feat)
        means = df_model[feat_options].mean().values

        x_range = np.linspace(df_model[x_feat].quantile(0.02), df_model[x_feat].quantile(0.98), grid_res)
        y_range = np.linspace(df_model[y_feat].quantile(0.02), df_model[y_feat].quantile(0.98), grid_res)
        XX, YY  = np.meshgrid(x_range, y_range)

        grid_input = np.tile(means, (grid_res * grid_res, 1))
        grid_input[:, xi] = XX.ravel()
        grid_input[:, yi] = YY.ravel()
        ZZ = pipe.predict(grid_input)
        if ZZ.ndim > 1:
            ZZ = ZZ.ravel()
        ZZ = ZZ.reshape(grid_res, grid_res)

        st.markdown(f"<span style='color:{PALETTE['muted']}'>Model R² on training data: <b style='color:{PALETTE['accent2']}'>{train_r2:.4f}</b> — all other features held at their dataset mean</span>", unsafe_allow_html=True)
        st.markdown("---")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Filled contour
        ax = axes[0]
        cf = ax.contourf(XX, YY, ZZ, levels=25, cmap=CMAP_CONTOUR, alpha=0.95)
        plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04, label=c_target)
        sc = ax.scatter(df_model[x_feat], df_model[y_feat],
                        c=df_model[c_target], cmap=CMAP_CONTOUR, s=15,
                        edgecolors=PALETTE["bg"], linewidth=0.3, alpha=0.85, zorder=5)
        ax.set_xlabel(x_feat, **LABEL_STYLE)
        ax.set_ylabel(y_feat, **LABEL_STYLE)
        ax.set_title(f"Filled contour\n{c_target}", **TITLE_STYLE)
        ax.grid(True, alpha=0.3)

        # Line contour with labels
        ax = axes[1]
        cs = ax.contour(XX, YY, ZZ, levels=15, cmap="coolwarm", linewidths=1.2)
        ax.clabel(cs, inline=True, fontsize=7, fmt="%.2f", colors=[PALETTE["text"]])
        ax.contourf(XX, YY, ZZ, levels=25, cmap=CMAP_CONTOUR, alpha=0.25)
        ax.set_xlabel(x_feat, **LABEL_STYLE)
        ax.set_ylabel(y_feat, **LABEL_STYLE)
        ax.set_title(f"Labelled contour\n{c_target}", **TITLE_STYLE)
        ax.grid(True, alpha=0.3)

        # 3D surface
        ax3d = fig.add_subplot(1, 3, 3, projection="3d")
        surf = ax3d.plot_surface(XX, YY, ZZ, cmap=CMAP_CONTOUR, edgecolor="none", alpha=0.9, rstride=2, cstride=2)
        ax3d.set_xlabel(x_feat, fontsize=7, color=PALETTE["text"], labelpad=4)
        ax3d.set_ylabel(y_feat, fontsize=7, color=PALETTE["text"], labelpad=4)
        ax3d.set_zlabel(c_target, fontsize=7, color=PALETTE["text"], labelpad=4)
        ax3d.set_title(f"3D response surface\n{c_target}", fontsize=10, color=PALETTE["text"], pad=6)
        ax3d.tick_params(labelsize=6, colors=PALETTE["muted"])
        ax3d.set_facecolor(PALETTE["panel"])
        fig.colorbar(surf, ax=ax3d, fraction=0.03, pad=0.04, shrink=0.5, label=c_target)

        plt.tight_layout(pad=1.5)
        fig_to_st(fig)

        # Marginal plots
        st.markdown("### Marginal effect plots")
        fig2, (ax_x, ax_y) = plt.subplots(1, 2, figsize=(12, 4))

        x_line = np.linspace(x_range[0], x_range[-1], 200)
        inp_x  = np.tile(means, (200, 1))
        inp_x[:, xi] = x_line
        z_x = pipe.predict(inp_x)
        if z_x.ndim > 1: z_x = z_x.ravel()
        ax_x.plot(x_line, z_x, color=PALETTE["accent1"], linewidth=2)
        ax_x.fill_between(x_line, z_x, alpha=0.15, color=PALETTE["accent1"])
        ax_x.set_xlabel(x_feat, **LABEL_STYLE)
        ax_x.set_ylabel(c_target, **LABEL_STYLE)
        ax_x.set_title(f"Marginal effect of {x_feat}", **TITLE_STYLE)
        ax_x.grid(True)

        y_line = np.linspace(y_range[0], y_range[-1], 200)
        inp_y  = np.tile(means, (200, 1))
        inp_y[:, yi] = y_line
        z_y = pipe.predict(inp_y)
        if z_y.ndim > 1: z_y = z_y.ravel()
        ax_y.plot(y_line, z_y, color=PALETTE["accent2"], linewidth=2)
        ax_y.fill_between(y_line, z_y, alpha=0.15, color=PALETTE["accent2"])
        ax_y.set_xlabel(y_feat, **LABEL_STYLE)
        ax_y.set_ylabel(c_target, **LABEL_STYLE)
        ax_y.set_title(f"Marginal effect of {y_feat}", **TITLE_STYLE)
        ax_y.grid(True)

        plt.tight_layout()
        fig_to_st(fig2)
