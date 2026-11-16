
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==========================================================
# CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Encrypted Traffic Security Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE = Path("/content/drive/MyDrive/EncryptedSDP")

DATA_PATH = BASE / "datasets/testing/raw/BCCC-CIRA-CIC-DoHBrw-2020.csv"
MODEL_PATH = BASE / "notebooks/random_forest_aligned_28f.pkl"

GREEN = "#0D9488"
RED = "#E45858"
BLUE = "#2563EB"
DARK = "#13243A"

# ==========================================================
# DASHBOARD STYLING
# ==========================================================

st.markdown("""
<style>
.stApp {
    background: #F3F7FC;
    color: #13243A;
}
[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1px solid #DCE5F0;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    color: #13243A !important;
}
.stMain h1, .stMain h2, .stMain h3,
.stMain p, .stMain label {
    color: #13243A !important;
}
[data-testid="stMetric"] {
    background: white;
    border: 1px solid #DEE7F1;
    border-radius: 14px;
    padding: 18px;
}
[data-testid="stMetricLabel"] *,
[data-testid="stMetricValue"] * {
    color: #13243A !important;
}
.stButton button,
.stFormSubmitButton button,
.stDownloadButton button {
    background: #2563EB !important;
    color: white !important;
    border-radius: 9px;
    border: none;
}
.stButton button p,
.stFormSubmitButton button p,
.stDownloadButton button p {
    color: white !important;
}
.hero {
    background: linear-gradient(115deg, #142C52, #2256A5);
    padding: 24px 28px;
    border-radius: 16px;
    margin-bottom: 20px;
}
.hero h2 {
    color: white !important;
    margin: 0;
}
.hero p {
    color: #DDEAFF !important;
    margin-top: 6px;
}
.check-card {
    background: white;
    border: 1px solid #DCE5F0;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 10px;
}
.check-card b {
    color: #13243A;
}
.check-card p {
    color: #475569;
    margin-bottom: 0;
}
.pass {
    float: right;
    color: #07846D;
    font-weight: 800;
}
.fail {
    float: right;
    color: #C83232;
    font-weight: 800;
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# REUSABLE UI FUNCTIONS
# ==========================================================

def hero(title, subtitle):
    st.markdown(
        f'<div class="hero"><h2>{title}</h2>'
        f'<p>{subtitle}</p></div>',
        unsafe_allow_html=True
    )


def metric_row(items):
    columns = st.columns(len(items))
    for column, (label, value) in zip(columns, items):
        column.metric(label, value)


def show_chart(fig):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(
            family="Arial",
            size=14,
            color="#172B4D"
        ),
        title=dict(
            font=dict(size=21, color="#102A43"),
            x=0.03
        ),
        xaxis=dict(
            title_font=dict(size=15, color="#172B4D"),
            tickfont=dict(size=12, color="#334155"),
            gridcolor="#E2E8F0"
        ),
        yaxis=dict(
            title_font=dict(size=15, color="#172B4D"),
            tickfont=dict(size=12, color="#334155"),
            gridcolor="#E2E8F0"
        ),
        legend=dict(
            font=dict(size=13, color="#172B4D"),
            title_font=dict(size=14, color="#102A43"),
            bgcolor="#FFFFFF"
        ),
        margin=dict(l=35, r=35, t=75, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)


# ==========================================================
# LOAD DATASET AND RANDOM FOREST MODEL
# ==========================================================

@st.cache_data
def load_data(path, modified):
    return pd.read_csv(path)


@st.cache_resource
def load_model(path, modified):
    return joblib.load(path)


if not DATA_PATH.exists():
    st.error("BCCC dataset not found.")
    st.stop()

if not MODEL_PATH.exists():
    st.error("Random Forest model not found.")
    st.stop()

full = load_data(
    str(DATA_PATH),
    DATA_PATH.stat().st_mtime
)

model = load_model(
    str(MODEL_PATH),
    MODEL_PATH.stat().st_mtime
)

if full.empty:
    st.error("Dataset is empty.")
    st.stop()

if not hasattr(model, "feature_names_in_"):
    st.error("Model feature names are unavailable.")
    st.stop()

FEATURES = list(model.feature_names_in_)

if len(FEATURES) != 28:
    st.error("Expected 28 model features.")
    st.stop()

missing = [f for f in FEATURES if f not in full.columns]

if missing:
    st.error(f"Missing model features: {missing}")
    st.stop()

TOTAL = len(full)

# ==========================================================
# SIDEBAR - DATASET SELECTION
# ==========================================================

st.sidebar.markdown("## Navigation Menu")
st.sidebar.markdown("### Dataset Selection")
st.sidebar.caption(f"Available BCCC records: {TOTAL:,}")

if "applied_selection" not in st.session_state:
    st.session_state.applied_selection = {
        "mode": "First N records",
        "start": 1,
        "end": min(5000, TOTAL),
        "count": min(5000, TOTAL),
        "seed": 42
    }

with st.sidebar.form("dataset_selection_form"):

    mode = st.selectbox(
        "Selection method",
        [
            "First N records",
            "Random N records",
            "Custom record range"
        ]
    )

    start = 1
    end = min(5000, TOTAL)
    count = min(5000, TOTAL)
    seed = 42

    if mode == "Custom record range":

        start = int(st.number_input(
            "Starting record",
            min_value=1,
            max_value=TOTAL,
            value=1,
            step=1
        ))

        end = int(st.number_input(
            "Ending record (inclusive)",
            min_value=start,
            max_value=TOTAL,
            value=min(start + 4999, TOTAL),
            step=1
        ))

        count = end - start + 1
        st.caption(f"Records to select: {count:,}")

    else:

        count = int(st.number_input(
            "Number of records",
            min_value=1,
            max_value=TOTAL,
            value=min(5000, TOTAL),
            step=1
        ))

        end = count

        if mode == "Random N records":

            seed = int(st.number_input(
                "Random seed",
                min_value=0,
                value=42,
                step=1
            ))

            st.caption(
                "The seed makes random selection reproducible."
            )

    apply_clicked = st.form_submit_button(
        "Apply Selection",
        type="primary",
        use_container_width=True
    )

if apply_clicked:
    st.session_state.applied_selection = {
        "mode": mode,
        "start": start,
        "end": end,
        "count": count,
        "seed": seed
    }

active = st.session_state.applied_selection

if active["mode"] == "Custom record range":

    selected = full.iloc[
        active["start"] - 1:active["end"]
    ].copy()

    selection_description = (
        f"Records {active['start']:,} to {active['end']:,}"
    )

elif active["mode"] == "Random N records":

    selected = full.sample(
        n=active["count"],
        random_state=active["seed"]
    ).sort_index().copy()

    selection_description = (
        f"{active['count']:,} random records "
        f"(seed {active['seed']})"
    )

else:

    selected = full.head(active["count"]).copy()
    selection_description = f"Records 1 to {active['count']:,}"

selected.insert(
    0,
    "Record ID",
    selected.index + 1
)

st.sidebar.success(f"Applied: {selection_description}")
st.sidebar.metric("Selected Records", f"{len(selected):,}")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select Module",
    [
        "1. Dataset Overview",
        "2. Dataset Validation",
        "3. Traffic Explorer",
        "4. Flow Behaviour Analysis",
        "5. Threat Detection Centre",
        "6. Explainable AI",
        "7. Model Performance",
        "8. Reports and Export"
    ]
)

# ==========================================================
# VALIDATE SELECTED RECORDS
# ==========================================================

X = selected[FEATURES].apply(
    pd.to_numeric,
    errors="coerce"
)

valid_mask = pd.Series(
    np.isfinite(X.to_numpy(dtype=float)).all(axis=1),
    index=selected.index
)

valid_data = selected.loc[valid_mask].copy()
X_valid = X.loc[valid_mask].copy()

# ==========================================================
# ACTUAL DATASET LABELS
# ==========================================================

def normalize_labels(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({
            "benign": 0,
            "malicious": 1,
            "0": 0,
            "1": 1
        })
    )


if "Label" in selected.columns:
    actual = normalize_labels(selected["Label"])
else:
    actual = pd.Series(np.nan, index=selected.index)

selected["Actual Class"] = actual.map({
    0: "Benign",
    1: "Malicious"
})

# ==========================================================
# RANDOM FOREST PREDICTIONS
# ==========================================================

@st.cache_data(show_spinner="Running Random Forest...")
def predict(_model, frame):
    if frame.empty:
        return np.array([], dtype=int)
    return np.asarray(_model.predict(frame), dtype=int)


predictions = predict(model, X_valid)

valid_data["Predicted ID"] = predictions

valid_data["Predicted Class"] = valid_data[
    "Predicted ID"
].map({
    0: "Benign",
    1: "Malicious"
})

valid_data["Actual ID"] = actual.loc[valid_data.index]

valid_data["Actual Class"] = valid_data[
    "Actual ID"
].map({
    0: "Benign",
    1: "Malicious"
})

evaluated = valid_data.dropna(
    subset=["Actual ID"]
).copy()

# ==========================================================
# CLASSIFICATION OUTCOMES
# ==========================================================

OUTCOME_NAMES = [
    "Benign → Benign (TN)",
    "Benign → Malicious (FP)",
    "Malicious → Malicious (TP)",
    "Malicious → Benign (FN)"
]

if not evaluated.empty:

    evaluated["Actual ID"] = evaluated["Actual ID"].astype(int)
    evaluated["Predicted ID"] = evaluated["Predicted ID"].astype(int)

    conditions = [
        (evaluated["Actual ID"] == 0) &
        (evaluated["Predicted ID"] == 0),

        (evaluated["Actual ID"] == 0) &
        (evaluated["Predicted ID"] == 1),

        (evaluated["Actual ID"] == 1) &
        (evaluated["Predicted ID"] == 1),

        (evaluated["Actual ID"] == 1) &
        (evaluated["Predicted ID"] == 0)
    ]

    evaluated["Prediction Outcome"] = np.select(
        conditions,
        OUTCOME_NAMES,
        default="Unknown"
    )

else:
    evaluated["Prediction Outcome"] = pd.Series(dtype=str)

OUTCOME_COUNTS = {
    name: int(
        (evaluated["Prediction Outcome"] == name).sum()
    )
    for name in OUTCOME_NAMES
}

# ==========================================================
# PERFORMANCE METRICS
# ==========================================================

if not evaluated.empty:

    y_true = evaluated["Actual ID"]
    y_pred = evaluated["Predicted ID"]

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = map(int, cm.ravel())

    scores = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "F1-Score": f1_score(
            y_true, y_pred, zero_division=0
        )
    }

else:
    cm = np.zeros((2, 2), dtype=int)
    tn = fp = fn = tp = 0
    scores = {}

# ==========================================================
# DISTRIBUTION CHART
# ==========================================================

def distribution_chart(data, column, title):

    counts = (
        data[column]
        .value_counts()
        .reindex(["Benign", "Malicious"], fill_value=0)
        .reset_index()
    )

    counts.columns = ["Class", "Records"]

    fig = px.bar(
        counts,
        x="Class",
        y="Records",
        color="Class",
        color_discrete_map={
            "Benign": GREEN,
            "Malicious": RED
        },
        text="Records",
        title=title
    )

    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False)

    return fig

# ==========================================================
# PAGE 1 - DATASET OVERVIEW
# ==========================================================

def page_overview():

    hero(
        "Dataset Overview",
        "Network traffic dataset and record selection"
    )

    metric_row([
        ("Available Records", f"{TOTAL:,}"),
        ("Selected Records", f"{len(selected):,}"),
        ("Valid Records", f"{valid_mask.sum():,}"),
        ("Model Features", "28")
    ])

    st.subheader("Applied Dataset Selection")

    st.info(
        f"{selection_description} | "
        f"Total selected: {len(selected):,}"
    )

    st.subheader("Dataset Information")
    st.write("Dataset: BCCC-CIRA-CIC-DoHBrw-2020")
    st.write("Classification Model: Random Forest")

    st.subheader("Selected Dataset Preview")

    st.dataframe(
        selected,
        use_container_width=True,
        hide_index=True,
        height=450
    )

# ==========================================================
# PAGE 2 - DATASET VALIDATION
# ==========================================================

def page_validation():

    hero(
        "Dataset Validation",
        "Validation results for the selected records"
    )

    passed = int(valid_mask.sum())
    failed = len(selected) - passed

    metric_row([
        ("Selected Records", f"{len(selected):,}"),
        ("Valid Records", f"{passed:,}"),
        ("Invalid Records", f"{failed:,}"),
        ("Validation Rate", f"{valid_mask.mean()*100:.2f}%")
    ])

    checks = [
        (
            "Dataset Source",
            "Expected BCCC dataset file",
            DATA_PATH.name == "BCCC-CIRA-CIC-DoHBrw-2020.csv"
        ),
        (
            "Feature Validation",
            "All 28 model features available",
            len(FEATURES) == 28 and not missing
        ),
        (
            "Missing Values",
            "No missing feature values",
            not X.isna().any().any()
        ),
        (
            "Numerical Integrity",
            "All feature values are finite",
            bool(valid_mask.all())
        ),
        (
            "Model Compatibility",
            "Correct feature order",
            list(X_valid.columns) == FEATURES
        )
    ]

    st.subheader("Validation Results")

    for name, description, passed_check in checks:

        status = (
            '<span class="pass">PASS</span>'
            if passed_check
            else '<span class="fail">FAIL</span>'
        )

        st.markdown(
            f'<div class="check-card">'
            f'<b>{name}</b>{status}'
            f'<p>{description}</p>'
            f'</div>',
            unsafe_allow_html=True
        )

    if all(item[2] for item in checks):
        st.success(
            f"All {len(selected):,} records passed "
            "the dataset integrity checks."
        )
    else:
        st.warning("Some validation checks failed.")

    with st.expander("View All 28 Features"):
        st.dataframe(
            pd.DataFrame({"Feature": FEATURES}),
            use_container_width=True,
            hide_index=True
        )

    st.caption(
        "CSV validation checks numerical integrity and "
        "model compatibility. It does not independently "
        "prove TLS encryption."
    )

# ==========================================================
# PAGE 3 - TRAFFIC EXPLORER
# ==========================================================

def page_explorer():

    hero(
        "Traffic Explorer",
        "Actual benign and malicious traffic distribution"
    )

    known = selected.dropna(subset=["Actual Class"])

    benign = int(
        (known["Actual Class"] == "Benign").sum()
    )

    malicious = int(
        (known["Actual Class"] == "Malicious").sum()
    )

    metric_row([
        ("Selected Records", f"{len(selected):,}"),
        ("Actual Benign", f"{benign:,}"),
        ("Actual Malicious", f"{malicious:,}")
    ])

    if known.empty:
        st.warning("No recognized actual labels.")
        return

    c1, c2 = st.columns(2)

    with c1:
        show_chart(
            distribution_chart(
                known,
                "Actual Class",
                "Actual Traffic Distribution"
            )
        )

    with c2:

        counts = (
            known["Actual Class"]
            .value_counts()
            .reset_index()
        )

        counts.columns = ["Class", "Records"]

        fig = px.pie(
            counts,
            names="Class",
            values="Records",
            hole=0.55,
            color="Class",
            color_discrete_map={
                "Benign": GREEN,
                "Malicious": RED
            },
            title="Traffic Composition"
        )

        show_chart(fig)

    st.subheader("Explore Traffic Records")

    filter_class = st.selectbox(
        "Filter Actual Traffic",
        ["All", "Benign", "Malicious"]
    )

    filtered = known.copy()

    if filter_class != "All":
        filtered = known[
            known["Actual Class"] == filter_class
        ]

    st.metric("Matching Records", f"{len(filtered):,}")

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True,
        height=450
    )

# ==========================================================
# PAGE 4 - FLOW BEHAVIOUR ANALYSIS
# ==========================================================

def page_flow():

    hero(
        "Flow Behaviour Analysis",
        "Statistical network-flow characteristics"
    )

    c1, c2 = st.columns(2)

    with c1:

        x_feature = st.selectbox(
            "X-axis Feature",
            FEATURES,
            index=(
                FEATURES.index("PacketLengthMean")
                if "PacketLengthMean" in FEATURES
                else 0
            )
        )

    with c2:

        y_feature = st.selectbox(
            "Y-axis Feature",
            FEATURES,
            index=(
                FEATURES.index("PacketTimeMean")
                if "PacketTimeMean" in FEATURES
                else 1
            )
        )

    plot_data = valid_data.copy()

    if len(plot_data) > 8000:
        plot_data = plot_data.sample(
            n=8000,
            random_state=42
        )

    if plot_data.empty:
        st.warning("No valid records to plot.")
        return

    fig = px.scatter(
        plot_data,
        x=x_feature,
        y=y_feature,
        color="Actual Class",
        color_discrete_map={
            "Benign": GREEN,
            "Malicious": RED
        },
        opacity=0.55,
        hover_data=["Record ID"],
        title="Network Flow Behaviour"
    )

    show_chart(fig)

    st.subheader("Feature Distribution")

    feature = st.selectbox(
        "Select Feature",
        FEATURES
    )

    fig2 = px.histogram(
        plot_data,
        x=feature,
        color="Actual Class",
        color_discrete_map={
            "Benign": GREEN,
            "Malicious": RED
        },
        nbins=45,
        barmode="overlay",
        title="Feature Distribution"
    )

    show_chart(fig2)

# ==========================================================
# PAGE 5 - THREAT DETECTION CENTRE
# ==========================================================

def page_threats():

    hero(
        "Threat Detection Centre",
        "Random Forest classification and record-level verification"
    )

    if valid_data.empty:
        st.warning("No valid records available.")
        return

    predicted_benign = int(
        (valid_data["Predicted ID"] == 0).sum()
    )

    predicted_malicious = int(
        (valid_data["Predicted ID"] == 1).sum()
    )

    metric_row([
        ("Records Classified", f"{len(valid_data):,}"),
        ("Predicted Benign", f"{predicted_benign:,}"),
        ("Predicted Malicious", f"{predicted_malicious:,}")
    ])

    st.subheader("Random Forest Predictions")

    show_chart(
        distribution_chart(
            valid_data,
            "Predicted Class",
            "Predicted Traffic Distribution"
        )
    )

    if evaluated.empty:
        st.warning("Actual labels unavailable for comparison.")
        return

    st.subheader("Prediction Verification")

    metric_row([
        (
            "Benign → Benign (TN)",
            f"{OUTCOME_COUNTS[OUTCOME_NAMES[0]]:,}"
        ),
        (
            "Benign → Malicious (FP)",
            f"{OUTCOME_COUNTS[OUTCOME_NAMES[1]]:,}"
        ),
        (
            "Malicious → Malicious (TP)",
            f"{OUTCOME_COUNTS[OUTCOME_NAMES[2]]:,}"
        ),
        (
            "Malicious → Benign (FN)",
            f"{OUTCOME_COUNTS[OUTCOME_NAMES[3]]:,}"
        )
    ])

    outcome_df = pd.DataFrame({
        "Outcome": [
            "Benign → Benign",
            "Benign → Malicious",
            "Malicious → Malicious",
            "Malicious → Benign"
        ],
        "Records": [
            OUTCOME_COUNTS[name]
            for name in OUTCOME_NAMES
        ]
    })

    fig = px.bar(
        outcome_df,
        x="Outcome",
        y="Records",
        color="Outcome",
        text="Records",
        color_discrete_map={
            "Benign → Benign": "#0D9488",
            "Benign → Malicious": "#F59E0B",
            "Malicious → Malicious": "#10B981",
            "Malicious → Benign": "#EF4444"
        },
        title="Prediction Verification Results"
    )

    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False)

    show_chart(fig)

    st.subheader("Inspect Records by Prediction Outcome")

    tab1, tab2, tab3, tab4 = st.tabs([
        f"Benign → Benign ({OUTCOME_COUNTS[OUTCOME_NAMES[0]]:,})",
        f"Benign → Malicious ({OUTCOME_COUNTS[OUTCOME_NAMES[1]]:,})",
        f"Malicious → Malicious ({OUTCOME_COUNTS[OUTCOME_NAMES[2]]:,})",
        f"Malicious → Benign ({OUTCOME_COUNTS[OUTCOME_NAMES[3]]:,})"
    ])

    tabs = [tab1, tab2, tab3, tab4]

    descriptions = [
        "Benign traffic correctly identified as benign.",
        "Benign traffic incorrectly identified as malicious.",
        "Malicious traffic correctly identified as malicious.",
        "Malicious traffic incorrectly identified as benign."
    ]

    display_columns = [
        "Record ID",
        "Actual Class",
        "Predicted Class",
        "Prediction Outcome"
    ]

    for i, tab in enumerate(tabs):

        with tab:

            category_data = evaluated[
                evaluated["Prediction Outcome"] == OUTCOME_NAMES[i]
            ].copy()

            st.write(descriptions[i])

            st.metric(
                "Matching Records",
                f"{len(category_data):,}"
            )

            st.dataframe(
                category_data[display_columns],
                use_container_width=True,
                hide_index=True,
                height=500
            )

            with st.expander("View All 28 Statistical Features"):

                st.dataframe(
                    category_data[display_columns + FEATURES],
                    use_container_width=True,
                    hide_index=True,
                    height=500
                )

            st.download_button(
                "Download These Records",
                data=category_data.to_csv(index=False),
                file_name=f"prediction_category_{i+1}.csv",
                mime="text/csv",
                key=f"download_category_{i}",
                use_container_width=True
            )

    correct = (
        OUTCOME_COUNTS[OUTCOME_NAMES[0]] +
        OUTCOME_COUNTS[OUTCOME_NAMES[2]]
    )

    incorrect = (
        OUTCOME_COUNTS[OUTCOME_NAMES[1]] +
        OUTCOME_COUNTS[OUTCOME_NAMES[3]]
    )

    st.subheader("Classification Summary")

    metric_row([
        ("Correct Predictions", f"{correct:,}"),
        ("Incorrect Predictions", f"{incorrect:,}")
    ])

# ==========================================================
# PAGE 6 - EXPLAINABLE AI (UPDATED)
# ==========================================================

def page_explainable():

    hero(
        "Explainable AI",
        "Random Forest feature importance analysis"
    )

    if not hasattr(model, "feature_importances_"):
        st.warning("Feature importance unavailable.")
        return

    importance = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": model.feature_importances_
    }).sort_values(
        "Importance",
        ascending=False
    ).reset_index(drop=True)

    st.subheader("Feature Importance Selection")

    top_n = st.slider(
        "Select Number of Important Features",
        min_value=1,
        max_value=len(FEATURES),
        value=min(10, len(FEATURES)),
        step=1
    )

    top = importance.head(top_n).copy()

    st.metric(
        "Features Displayed",
        f"{top_n} / {len(FEATURES)}"
    )

    fig = px.bar(
        top.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Blues",
        title=f"Top {top_n} Important Features"
    )

    fig.update_layout(
        height=max(400, top_n * 32),
        coloraxis_showscale=False
    )

    show_chart(fig)

    st.subheader("Selected Feature Rankings")

    top_display = top.copy()
    top_display.insert(
        0,
        "Rank",
        range(1, len(top_display) + 1)
    )

    top_display["Importance (%)"] = (
        top_display["Importance"] * 100
    ).round(4)

    st.dataframe(
        top_display[
            ["Rank", "Feature", "Importance (%)"]
        ],
        use_container_width=True,
        hide_index=True
    )

    with st.expander("View Importance of All 28 Features"):

        all_display = importance.copy()

        all_display.insert(
            0,
            "Rank",
            range(1, len(all_display) + 1)
        )

        all_display["Importance (%)"] = (
            all_display["Importance"] * 100
        ).round(4)

        st.dataframe(
            all_display[
                ["Rank", "Feature", "Importance (%)"]
            ],
            use_container_width=True,
            hide_index=True
        )

    st.caption(
        "Feature importance is obtained from the trained "
        "Random Forest model. Changing the slider changes "
        "the number of features displayed, not the model."
    )

# ==========================================================
# PAGE 7 - MODEL PERFORMANCE
# ==========================================================

def page_performance():

    hero(
        "Model Performance",
        "Random Forest classification evaluation"
    )

    if evaluated.empty:
        st.warning("No labeled records available.")
        return

    metric_row([
        (name, f"{value * 100:.2f}%")
        for name, value in scores.items()
    ])

    fig = go.Figure(
        go.Heatmap(
            z=cm,
            x=["Benign", "Malicious"],
            y=["Benign", "Malicious"],
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 22},
            colorscale="Blues"
        )
    )

    fig.update_layout(
        title="Random Forest Confusion Matrix",
        xaxis_title="Predicted Class",
        yaxis_title="Actual Class",
        height=430
    )

    fig.update_yaxes(autorange="reversed")

    show_chart(fig)

    st.subheader("Prediction Outcomes")

    metric_row([
        ("True Positive", f"{tp:,}"),
        ("True Negative", f"{tn:,}"),
        ("False Positive", f"{fp:,}"),
        ("False Negative", f"{fn:,}")
    ])

    st.subheader("Classification Report")

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Benign", "Malicious"],
        zero_division=0
    )

    st.code(report, language="text")

    st.caption(
        "Metrics are calculated for the selected labeled "
        "records. Independent test performance requires "
        "checking for training and testing overlap."
    )

# ==========================================================
# PAGE 8 - REPORTS AND EXPORT
# ==========================================================

def page_reports():

    hero(
        "Reports and Export",
        "Consolidated Random Forest traffic analysis"
    )

    summary = {
        "Dataset": "BCCC-CIRA-CIC-DoHBrw-2020",
        "Selection": selection_description,
        "Available Records": TOTAL,
        "Selected Records": len(selected),
        "Valid Records": int(valid_mask.sum()),
        "Invalid Records": int((~valid_mask).sum()),
        "Actual Benign": int((actual == 0).sum()),
        "Actual Malicious": int((actual == 1).sum()),
        "Predicted Benign": int(
            (valid_data["Predicted ID"] == 0).sum()
        ),
        "Predicted Malicious": int(
            (valid_data["Predicted ID"] == 1).sum()
        ),
        "True Positive": tp,
        "True Negative": tn,
        "False Positive": fp,
        "False Negative": fn
    }

    summary.update(scores)

    st.dataframe(
        pd.DataFrame(
            summary.items(),
            columns=["Metric", "Value"]
        ),
        use_container_width=True,
        hide_index=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.download_button(
            "All Prediction Results",
            data=valid_data.to_csv(index=False),
            file_name="all_predictions.csv",
            mime="text/csv"
        )

    with c2:
        threats = valid_data[
            valid_data["Predicted ID"] == 1
        ]

        st.download_button(
            "Malicious Traffic Report",
            data=threats.to_csv(index=False),
            file_name="malicious_traffic.csv",
            mime="text/csv"
        )

    with c3:
        st.download_button(
            "Summary Report",
            data=pd.Data([summary]).to_csv(index=False),
            file_name="summary_report.csv",
            mime="text/csv"
        )

# ==========================================================
# PAGE NAVIGATION
# ==========================================================

if page == "1. Dataset Overview":
    page_overview()

elif page == "2. Dataset Validation":
    page_validation()

elif page == "3. Traffic Explorer":
    page_explorer()

elif page == "4. Flow Behaviour Analysis":
    page_flow()

elif page == "5. Threat Detection Centre":
    page_threats()

elif page == "6. Explainable AI":
    page_explainable()

elif page == "7. Model Performance":
    page_performance()

elif page == "8. Reports and Export":
    page_reports()
