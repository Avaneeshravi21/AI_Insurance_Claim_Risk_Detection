import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

try:
    import matplotlib
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from dotenv import load_dotenv
import os

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    try:
        gemini_api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        gemini_api_key = None

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Insurance Claim Risk Detection",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "random_forest_pipeline.pkl"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    loaded_object = joblib.load(MODEL_PATH)

    return loaded_object


try:

    loaded_pipeline = load_model()

except Exception as e:

    st.error(
        f"Unable to load model: {e}"
    )

    st.stop()


# ============================================================
# SUPPORT BOTH SAVED MODEL FORMATS
# ============================================================

random_forest_pipeline = None
random_forest_model = None
preprocessor = None


if isinstance(loaded_pipeline, dict):

    # Your current saved object is expected to be:
    # {
    #     "preprocessor": ...,
    #     "model": ...
    # }

    if "preprocessor" in loaded_pipeline:
        preprocessor = loaded_pipeline["preprocessor"]

    if "model" in loaded_pipeline:
        random_forest_model = loaded_pipeline["model"]

    # Recreate a pipeline-like object only when possible
    if (
        preprocessor is not None
        and random_forest_model is not None
    ):

        from sklearn.pipeline import Pipeline

        random_forest_pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    random_forest_model
                )
            ]
        )

else:

    # Normal sklearn Pipeline
    random_forest_pipeline = loaded_pipeline

    try:

        preprocessor = (
            random_forest_pipeline[
                "preprocessor"
            ]
        )

    except Exception:

        preprocessor = None

    try:

        random_forest_model = (
            random_forest_pipeline[
                "model"
            ]
        )

    except Exception:

        random_forest_model = loaded_pipeline


# ============================================================
# CHECK MODEL
# ============================================================

if random_forest_model is None:

    st.error(
        "Random Forest model could not be found "
        "inside the saved model file."
    )

    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title(
    "🛡️ AI Insurance Claim Risk Detection"
)

st.write(
    "Predict insurance claim fraud risk using a trained "
    "Random Forest Machine Learning model."
)


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander("🤖 Model Information"):

    st.write(
        "Model: Random Forest Classifier"
    )

    st.write(
        "Preprocessing: SimpleImputer + StandardScaler"
    )

    st.write(
        "Model input: 25 processed features"
    )

    st.write(
        "Training data: Insurance claims dataset"
    )


# ============================================================
# REQUIRED MODEL FEATURES
# ============================================================

NUMERIC_FEATURES = [
    "Customer_Age",
    "Policy_Tenure_Years",
    "Claim_Amount",
    "Previous_Claim_Count",
    "Vehicle_Age",
    "Repair_Estimate",
    "Final_Invoice_Amount",
    "Submission_Delay_Days",
    "Invoice_Variance_Percentage",
    "Missing_Information_Count",
    "Incident_Year",
    "Incident_Month",
    "Incident_Day",
    "Incident_DayOfWeek",
    "Submission_Year",
    "Submission_Month",
    "Submission_Day",
    "Submission_DayOfWeek",
    "Description_Length",
    "Description_Word_Count",
    "Claim_Repair_Ratio",
    "Claims_Per_Tenure_Year",
    "High_Value_Claim"
]

BINARY_FEATURES = [
    "Police_Report_Available",
    "Witness_Available"
]

MODEL_FEATURES = (
    NUMERIC_FEATURES +
    BINARY_FEATURES
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def convert_binary_column(series):

    """
    Convert common Yes/No, True/False and 1/0
    values into numeric 1/0.
    """

    if pd.api.types.is_numeric_dtype(series):

        return pd.to_numeric(
            series,
            errors="coerce"
        )

    cleaned = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )

    mapping = {
        "yes": 1,
        "y": 1,
        "true": 1,
        "1": 1,
        "available": 1,
        "present": 1,

        "no": 0,
        "n": 0,
        "false": 0,
        "0": 0,
        "not available": 0,
        "unavailable": 0,
        "absent": 0
    }

    return cleaned.map(mapping)


def prepare_model_input(df):

    """
    Convert the original raw insurance claim dataset
    into the exact feature structure expected by the
    trained Random Forest model.
    """

    data = df.copy()


    # --------------------------------------------------------
    # 1. Convert numeric columns
    # --------------------------------------------------------

    numeric_base_columns = [
        "Customer_Age",
        "Policy_Tenure_Years",
        "Claim_Amount",
        "Previous_Claim_Count",
        "Vehicle_Age",
        "Repair_Estimate",
        "Final_Invoice_Amount",
        "Submission_Delay_Days",
        "Invoice_Variance_Percentage",
        "Missing_Information_Count"
    ]

    for column in numeric_base_columns:

        if column in data.columns:

            data[column] = pd.to_numeric(
                data[column],
                errors="coerce"
            )


    # --------------------------------------------------------
    # 2. Convert binary columns
    # --------------------------------------------------------

    for column in BINARY_FEATURES:

        if column in data.columns:

            data[column] = convert_binary_column(
                data[column]
            )

        else:

            data[column] = 0


    # --------------------------------------------------------
    # 3. Incident Date features
    # --------------------------------------------------------

    if "Incident_Date" in data.columns:

        incident_date = pd.to_datetime(
            data["Incident_Date"],
            errors="coerce",
            dayfirst=True
        )

    else:

        incident_date = pd.Series(
            pd.NaT,
            index=data.index
        )


    data["Incident_Year"] = (
        incident_date.dt.year
    )

    data["Incident_Month"] = (
        incident_date.dt.month
    )

    data["Incident_Day"] = (
        incident_date.dt.day
    )

    data["Incident_DayOfWeek"] = (
        incident_date.dt.dayofweek
    )


    # --------------------------------------------------------
    # 4. Submission Date features
    # --------------------------------------------------------

    if "Claim_Submission_Date" in data.columns:

        submission_date = pd.to_datetime(
            data["Claim_Submission_Date"],
            errors="coerce",
            dayfirst=True
        )

    else:

        submission_date = pd.Series(
            pd.NaT,
            index=data.index
        )


    data["Submission_Year"] = (
        submission_date.dt.year
    )

    data["Submission_Month"] = (
        submission_date.dt.month
    )

    data["Submission_Day"] = (
        submission_date.dt.day
    )

    data["Submission_DayOfWeek"] = (
        submission_date.dt.dayofweek
    )


    # --------------------------------------------------------
    # 4b. Submission Delay (days between incident and submission)
    # --------------------------------------------------------

    data["Submission_Delay_Days"] = (
        (submission_date - incident_date).dt.days
    )


    # --------------------------------------------------------
    # 5. Customer description features
    # --------------------------------------------------------

    if "Customer_Description" in data.columns:

        description = (
            data["Customer_Description"]
            .fillna("")
            .astype(str)
        )

    else:

        description = pd.Series(
            "",
            index=data.index
        )


    data["Description_Length"] = (
        description.str.len()
    )

    data["Description_Word_Count"] = (
        description.str.split().str.len()
    )


    # --------------------------------------------------------
    # 6. Claim / Repair Ratio
    # --------------------------------------------------------

    claim_amount = pd.to_numeric(
        data["Claim_Amount"],
        errors="coerce"
    )

    repair_estimate = pd.to_numeric(
        data["Repair_Estimate"],
        errors="coerce"
    )

    data["Claim_Repair_Ratio"] = np.where(
        claim_amount > 0,
        repair_estimate / claim_amount,
        0
    )


    # --------------------------------------------------------
    # 6b. Invoice Variance Percentage
    # (how much the final invoice overshot the repair estimate)
    # --------------------------------------------------------

    final_invoice_amount = pd.to_numeric(
        data["Final_Invoice_Amount"],
        errors="coerce"
    )

    data["Invoice_Variance_Percentage"] = np.where(
        repair_estimate > 0,
        (final_invoice_amount - repair_estimate) / repair_estimate * 100,
        0
    )


    # --------------------------------------------------------
    # 7. Claims per tenure year
    # --------------------------------------------------------

    previous_claims = pd.to_numeric(
        data["Previous_Claim_Count"],
        errors="coerce"
    )

    tenure = pd.to_numeric(
        data["Policy_Tenure_Years"],
        errors="coerce"
    )

    data["Claims_Per_Tenure_Year"] = np.where(
        tenure > 0,
        previous_claims / tenure,
        previous_claims
    )


    # --------------------------------------------------------
    # 8. High Value Claim
    #
    # Use claim amount >= 75th percentile of uploaded data.
    # This creates a numeric 0/1 feature.
    # --------------------------------------------------------

    if "Claim_Amount" in data.columns:

        try:

            high_value_threshold = (
                data["Claim_Amount"]
                .quantile(0.75)
            )

            data["High_Value_Claim"] = (
                data["Claim_Amount"]
                >= high_value_threshold
            ).astype(int)

        except Exception:

            data["High_Value_Claim"] = 0

    else:

        data["High_Value_Claim"] = 0


    # --------------------------------------------------------
    # 8b. Missing Information Count
    # (how many of the original raw fields were left blank)
    # --------------------------------------------------------

    raw_fields_to_check = [
        col for col in [
            "Customer_Age", "Policy_Type", "Policy_Tenure_Years",
            "Claim_Amount", "Incident_Type", "Incident_Date",
            "Claim_Submission_Date", "Previous_Claim_Count",
            "Vehicle_Age", "Repair_Estimate", "Final_Invoice_Amount",
            "Police_Report_Available", "Witness_Available", "Location"
        ]
        if col in df.columns
    ]

    data["Missing_Information_Count"] = (
        df[raw_fields_to_check].isna().sum(axis=1)
    )


    # --------------------------------------------------------
    # 9. Make sure every required feature exists
    # --------------------------------------------------------

    for column in MODEL_FEATURES:

        if column not in data.columns:

            data[column] = np.nan


    # --------------------------------------------------------
    # 10. Select ONLY model features
    # --------------------------------------------------------

    model_input = data[
        MODEL_FEATURES
    ].copy()


    # --------------------------------------------------------
    # 11. Force numeric features to numeric
    # --------------------------------------------------------

    for column in NUMERIC_FEATURES:

        model_input[column] = pd.to_numeric(
            model_input[column],
            errors="coerce"
        )


    # --------------------------------------------------------
    # 12. Force binary features to numeric
    # --------------------------------------------------------

    for column in BINARY_FEATURES:

        model_input[column] = pd.to_numeric(
            model_input[column],
            errors="coerce"
        )


    return model_input


def get_risk_category(probability):

    if probability >= 0.60:

        return "High"

    elif probability >= 0.30:

        return "Medium"

    else:

        return "Low"


def draw_risk_gauge(probability, title="Risk Level"):
    """
    Draws a speedometer-style gauge (green -> amber -> red) with a
    needle pointing at the given probability. Used for both the
    portfolio-level average risk and an individual claim's risk,
    so risk reads as a visual "dial" instead of a raw number.
    """

    if not MATPLOTLIB_AVAILABLE:
        return None

    probability = max(0.0, min(1.0, float(probability)))

    fig, ax = plt.subplots(figsize=(4, 2.6), subplot_kw={"aspect": "equal"})

    zones = [
        (0.0, 0.30, "#2E7D32"),   # Low - green
        (0.30, 0.60, "#F9A825"),  # Medium - amber
        (0.60, 1.0, "#C62828"),   # High - red
    ]

    for start, end, color in zones:
        theta1 = 180 - (start * 180)
        theta2 = 180 - (end * 180)
        wedge = matplotlib.patches.Wedge(
            (0, 0), 1.0, theta2, theta1,
            width=0.3, facecolor=color, edgecolor="white"
        )
        ax.add_patch(wedge)

    needle_angle = np.radians(180 - probability * 180)
    x = 0.85 * np.cos(needle_angle)
    y = 0.85 * np.sin(needle_angle)
    ax.plot([0, x], [0, y], color="black", linewidth=2.5, solid_capstyle="round")
    ax.add_patch(plt.Circle((0, 0), 0.05, color="black"))

    ax.text(0, -0.25, f"{probability:.0%}", ha="center", va="center",
            fontsize=18, fontweight="bold")
    ax.text(0, -0.45, title, ha="center", va="center",
            fontsize=10, color="#555555")

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.5, 1.1)
    ax.axis("off")

    return fig


FRIENDLY_FACTOR_MAP = {
    "Customer_Age": "your age",
    "Policy_Tenure_Years": "how long you've held your policy",
    "Claim_Amount": "the amount claimed",
    "Previous_Claim_Count": "your past claim history",
    "Vehicle_Age": "your vehicle's age",
    "Repair_Estimate": "the repair cost estimate",
    "Final_Invoice_Amount": "the final invoice amount",
    "Submission_Delay_Days": "how quickly the claim was submitted",
    "Invoice_Variance_Percentage": "the difference between the estimate and the final invoice",
    "Missing_Information_Count": "how complete the submitted information was",
    "Claim_Repair_Ratio": "how the claim amount compares to the repair estimate",
    "Claims_Per_Tenure_Year": "your overall claim frequency",
    "High_Value_Claim": "the overall size of this claim",
    "Police_Report_Available": "whether a police report was provided",
    "Witness_Available": "whether a witness was available",
    "Description_Length": "the detail level of the claim description",
    "Description_Word_Count": "the detail level of the claim description",
}


def humanize_feature_name(raw_feature_name):
    """
    Converts a processed/prefixed feature name (e.g. 'num__Invoice_Variance_Percentage'
    or 'cat__Policy_Type_Comprehensive') into a plain-English phrase a
    customer can understand, using FRIENDLY_FACTOR_MAP where possible.
    """

    # Strip common ColumnTransformer prefixes like "num__", "cat__", "binary__"
    cleaned = raw_feature_name
    if "__" in cleaned:
        cleaned = cleaned.split("__", 1)[1]

    # Try an exact match first
    if cleaned in FRIENDLY_FACTOR_MAP:
        return FRIENDLY_FACTOR_MAP[cleaned]

    # Try matching the base feature name if this is a one-hot encoded column
    # (e.g. "Policy_Type_Comprehensive" -> base "Policy_Type")
    for base_name, phrase in FRIENDLY_FACTOR_MAP.items():
        if cleaned.startswith(base_name):
            return phrase

    # Fallback: turn "Some_Feature_Name" into "some feature name"
    return cleaned.replace("_", " ").lower()


def technical_feature_label(raw_feature_name):
    """
    Converts a processed/prefixed feature name into a clean TECHNICAL
    label for investigator-facing charts (e.g. 'Customer_Age' instead
    of the customer-facing phrase 'your age'). Uses the same base-name
    matching as humanize_feature_name so one-hot encoded columns still
    group together correctly (e.g. 'Policy_Type_Comprehensive' and
    'Policy_Type_Third-Party' both group under 'Policy Type').
    """

    cleaned = raw_feature_name
    if "__" in cleaned:
        cleaned = cleaned.split("__", 1)[1]

    if cleaned in FRIENDLY_FACTOR_MAP:
        return cleaned.replace("_", " ")

    for base_name in FRIENDLY_FACTOR_MAP:
        if cleaned.startswith(base_name):
            return base_name.replace("_", " ")

    return cleaned.replace("_", " ")


# ============================================================
# CSV UPLOAD
# ============================================================

st.header(
    "📂 Upload Insurance Claims CSV"
)

uploaded_file = st.file_uploader(
    "Upload your insurance claims CSV file",
    type=["csv"]
)


# ============================================================
# PROCESS ONLY AFTER FILE UPLOAD
# ============================================================

if uploaded_file is None:

    st.info(
        "Please upload your insurance claims CSV file "
        "to start model scoring."
    )

    st.stop()


# ============================================================
# READ CSV
# ============================================================

try:

    uploaded_df = pd.read_csv(
        uploaded_file
    )

except Exception as e:

    st.error(
        f"Unable to read CSV: {e}"
    )

    st.stop()


st.success(
    "CSV uploaded successfully!"
)


# ============================================================
# DATASET INFORMATION
# ============================================================

st.write(
    "Original Dataset Shape:",
    uploaded_df.shape
)

st.dataframe(
    uploaded_df.head(10),
    use_container_width=True
)


# ============================================================
# DATA QUALITY
# ============================================================

st.header(
    "🔍 Data Quality Information"
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Rows",
        uploaded_df.shape[0]
    )

with col2:

    st.metric(
        "Columns",
        uploaded_df.shape[1]
    )

with col3:

    st.metric(
        "Missing Values",
        int(
            uploaded_df
            .isna()
            .sum()
            .sum()
        )
    )

with col4:

    st.metric(
        "Duplicate Rows",
        int(
            uploaded_df
            .duplicated()
            .sum()
        )
    )


with st.expander(
    "📊 Missing Values by Column"
):

    missing_df = (
        uploaded_df
        .isna()
        .sum()
        .reset_index()
    )

    missing_df.columns = [
        "Column",
        "Missing Values"
    ]

    st.dataframe(
        missing_df,
        use_container_width=True
    )


# ============================================================
# PREPARE MODEL INPUT
# ============================================================

st.header(
    "⚙️ Preparing Data for Model"
)

try:

    model_input = prepare_model_input(
        uploaded_df
    )

    st.success(
        "Raw claim data successfully converted "
        "into model input features."
    )

    st.write(
        "Model Input Shape:",
        model_input.shape
    )

except Exception as e:

    st.error(
        f"Data preparation error: {e}"
    )

    st.stop()


# ============================================================
# VERIFY FEATURES
# ============================================================

missing_model_features = [
    column
    for column in MODEL_FEATURES
    if column not in model_input.columns
]

if len(missing_model_features) > 0:

    st.error(
        "Required model features are missing:"
    )

    st.write(
        missing_model_features
    )

    st.stop()


# ============================================================
# MODEL SCORING
# ============================================================

st.header(
    "🔮 Model Scoring"
)

try:

    # --------------------------------------------------------
    # Check fitted status
    # --------------------------------------------------------

    if not hasattr(
        random_forest_model,
        "classes_"
    ):

        st.error(
            "The Random Forest model loaded from the "
            "PKL file is not fitted."
        )

        st.stop()


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    all_predictions = (
        random_forest_pipeline
        .predict(
            model_input
        )
    )

    all_probabilities = (
        random_forest_pipeline
        .predict_proba(
            model_input
        )[:, 1]
    )


    # --------------------------------------------------------
    # Create final scored dataset
    # --------------------------------------------------------

    final_scored_df = (
        uploaded_df.copy()
    )


    final_scored_df[
        "Model_Prediction"
    ] = all_predictions


    final_scored_df[
        "Risk_Probability"
    ] = all_probabilities


    final_scored_df[
        "Risk_Category"
    ] = [
        get_risk_category(
            probability
        )
        for probability
        in all_probabilities
    ]


    st.success(
        f"Prediction completed successfully for "
        f"{len(final_scored_df)} claims."
    )


    # --------------------------------------------------------
    # Save a shared copy so the separate Customer page can look
    # up any claim by ID later, in a different browser session,
    # without needing to re-upload or re-score anything.
    # --------------------------------------------------------

    try:

        shared_data_dir = os.path.join(BASE_DIR, "shared_data")
        os.makedirs(shared_data_dir, exist_ok=True)

        # Save ALL model-feature columns under a distinct "MF__" prefix,
        # regardless of whether a same-named raw column also exists in
        # final_scored_df. This guarantees the engineered/encoded values
        # (e.g. Police_Report_Available as 0/1) are never confused with
        # the raw display version (e.g. Police_Report_Available as
        # "Yes"/"No") when this file is read back later.
        model_feature_export = model_input[MODEL_FEATURES].reset_index(drop=True)
        model_feature_export.columns = [f"MF__{c}" for c in model_feature_export.columns]

        shared_export_df = pd.concat(
            [
                final_scored_df.reset_index(drop=True),
                model_feature_export,
            ],
            axis=1,
        )

        shared_export_df.to_csv(
            os.path.join(shared_data_dir, "latest_scored_claims.csv"),
            index=False,
        )

    except Exception as shared_save_error:

        st.warning(
            f"Could not save shared claims data for the Customer "
            f"page: {shared_save_error}"
        )


except Exception as e:

    st.error(
        f"Model scoring error: {e}"
    )

    st.write(
        "Model input shape:",
        model_input.shape
    )

    st.write(
        "Expected model features:",
        len(MODEL_FEATURES)
    )

    st.stop()


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.header(
    "📊 Model Performance"
)


target_column = None

possible_targets = [
    "Fraud_Label",
    "Fraud",
    "Fraud_Flag",
    "Is_Fraud",
    "Fraud or Risk_Label"
]


for possible_target in possible_targets:

    if possible_target in uploaded_df.columns:

        target_column = possible_target

        break


if target_column is not None:

    try:

        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            confusion_matrix
        )


        actual = (
            uploaded_df[
                target_column
            ]
            .copy()
        )


        # ----------------------------------------------------
        # Convert target
        # ----------------------------------------------------

        if actual.dtype == "object":

            cleaned_target = (
                actual
                .astype(str)
                .str.strip()
                .str.lower()
            )

            target_mapping = {

                "fraud": 1,
                "yes": 1,
                "true": 1,
                "1": 1,
                "high": 1,

                "not fraud": 0,
                "no": 0,
                "false": 0,
                "0": 0,
                "low": 0
            }

            actual = cleaned_target.map(
                target_mapping
            )


        actual = pd.to_numeric(
            actual,
            errors="coerce"
        )


        valid_mask = actual.notna()


        actual_valid = (
            actual[
                valid_mask
            ]
            .astype(int)
            .values
        )


        prediction_valid = (
            np.asarray(
                all_predictions
            )[valid_mask.values]
        )


        if len(actual_valid) > 0:

            metric1, metric2, metric3, metric4 = (
                st.columns(4)
            )


            with metric1:

                st.metric(
                    "Accuracy",
                    f"{accuracy_score(actual_valid, prediction_valid):.2%}"
                )


            with metric2:

                st.metric(
                    "Precision",
                    f"{precision_score(actual_valid, prediction_valid, zero_division=0):.2%}"
                )


            with metric3:

                st.metric(
                    "Recall",
                    f"{recall_score(actual_valid, prediction_valid, zero_division=0):.2%}"
                )


            with metric4:

                st.metric(
                    "F1 Score",
                    f"{f1_score(actual_valid, prediction_valid, zero_division=0):.2%}"
                )


            st.subheader(
                "Confusion Matrix"
            )


            cm = confusion_matrix(
                actual_valid,
                prediction_valid
            )


            cm_df = pd.DataFrame(
                cm
            )


            st.dataframe(
                cm_df,
                use_container_width=True
            )


    except Exception as e:

        st.warning(
            f"Performance calculation unavailable: {e}"
        )


else:

    st.info(
        "Fraud target column was not found. "
        "Performance metrics cannot be calculated "
        "for this uploaded file."
    )


# ============================================================
# RISK SUMMARY
# ============================================================

st.header(
    "📈 Risk Category Summary"
)


risk_summary = (
    final_scored_df[
        "Risk_Category"
    ]
    .value_counts()
)


low_count = int(
    risk_summary.get(
        "Low",
        0
    )
)

medium_count = int(
    risk_summary.get(
        "Medium",
        0
    )
)

high_count = int(
    risk_summary.get(
        "High",
        0
    )
)


risk_col1, risk_col2, risk_col3 = (
    st.columns(3)
)


total_claims = low_count + medium_count + high_count

with risk_col1:

    st.metric(
        "🟢 Low Risk",
        low_count,
        f"{(low_count / total_claims * 100 if total_claims else 0):.1f}% of claims"
    )


with risk_col2:

    st.metric(
        "🟡 Medium Risk",
        medium_count,
        f"{(medium_count / total_claims * 100 if total_claims else 0):.1f}% of claims"
    )


with risk_col3:

    st.metric(
        "🔴 High Risk",
        high_count,
        f"{(high_count / total_claims * 100 if total_claims else 0):.1f}% of claims"
    )


# ============================================================
# INTERACTIVE RISK DASHBOARD
# ============================================================

st.subheader(
    "📊 Risk Dashboard"
)

RISK_COLORS = {
    "Low": "#2E7D32",      # green
    "Medium": "#F9A825",   # amber
    "High": "#C62828",     # red
}

risk_order = ["Low", "Medium", "High"]
risk_counts_ordered = [
    low_count,
    medium_count,
    high_count,
]
colors_ordered = [RISK_COLORS[r] for r in risk_order]


# ---- Customize Dashboard: choose which panels to show ----

with st.expander("⚙️ Customize Dashboard", expanded=False):

    customize_col1, customize_col2, customize_col3 = st.columns(3)

    with customize_col1:
        show_donut = st.checkbox("Risk Distribution", value=True)
        show_breakdown = st.checkbox("Category Breakdown", value=True)

    with customize_col2:
        show_scatter = st.checkbox("Amount vs Risk Scatter", value=True)
        show_histogram = st.checkbox("Risk Score Distribution", value=True)

    with customize_col3:
        show_leaderboard = st.checkbox("Top 10 Riskiest Claims", value=True)
        show_factors = st.checkbox("Top Risk Factors", value=True)


# ---- Dashboard controls: these two widgets drive every panel below ----

control_col1, control_col2 = st.columns(2)

with control_col1:

    risk_focus = st.radio(
        "Focus on risk category",
        options=["All"] + risk_order,
        horizontal=True,
        help="Every chart and table below updates to match your selection.",
    )

with control_col2:

    breakdown_candidates = [
        col for col in
        ["Incident_Type", "Policy_Type", "Location",
         "Police_Report_Available", "Witness_Available"]
        if col in final_scored_df.columns
    ]

    breakdown_dimension = st.selectbox(
        "Break down by",
        options=breakdown_candidates if breakdown_candidates else ["Risk_Category"],
        help="Controls the Category Breakdown chart below.",
    )


# ---- This single filtered dataframe powers every panel AND the table further down ----

if risk_focus == "All":
    dashboard_filtered_df = final_scored_df.copy()
else:
    dashboard_filtered_df = final_scored_df[
        final_scored_df["Risk_Category"] == risk_focus
    ].copy()

st.caption(
    f"Showing **{len(dashboard_filtered_df)}** of {len(final_scored_df)} claims "
    f"({risk_focus} risk focus)"
)


# ---- Row 1: Risk Distribution (donut) + Category Breakdown ----

if show_donut or show_breakdown:

    dash_row1_col1, dash_row1_col2 = st.columns(2)

    if show_donut:

        with dash_row1_col1:

            st.caption("Risk Distribution")

            if PLOTLY_AVAILABLE:

                donut_data = pd.DataFrame({
                    "Risk Category": risk_order,
                    "Claims": risk_counts_ordered,
                })
                donut_data = donut_data[donut_data["Claims"] > 0]

                if len(donut_data) > 0:
                    fig_donut = px.pie(
                        donut_data,
                        names="Risk Category",
                        values="Claims",
                        hole=0.55,
                        color="Risk Category",
                        color_discrete_map=RISK_COLORS,
                    )
                    fig_donut.update_traces(
                        textinfo="label+percent",
                        textposition="outside",
                    )
                    fig_donut.update_layout(
                        showlegend=False,
                        margin=dict(t=10, b=10, l=10, r=10),
                        height=320,
                    )
                    st.plotly_chart(fig_donut, use_container_width=True)

            elif MATPLOTLIB_AVAILABLE:

                nonzero_labels = [
                    f"{label} ({count})"
                    for label, count in zip(risk_order, risk_counts_ordered)
                    if count > 0
                ]
                nonzero_values = [c for c in risk_counts_ordered if c > 0]
                nonzero_colors = [
                    RISK_COLORS[label]
                    for label, count in zip(risk_order, risk_counts_ordered)
                    if count > 0
                ]
                fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
                if len(nonzero_values) > 0:
                    ax_pie.pie(
                        nonzero_values, labels=nonzero_labels,
                        colors=nonzero_colors, autopct="%1.1f%%", startangle=90,
                        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
                    )
                st.pyplot(fig_pie)
                plt.close(fig_pie)

            else:

                st.bar_chart(
                    pd.DataFrame({"Risk Category": risk_order, "Claims": risk_counts_ordered})
                    .set_index("Risk Category")
                )

    if show_breakdown:

        with dash_row1_col2:

            st.caption(f"Breakdown by {breakdown_dimension}")

            if breakdown_dimension in dashboard_filtered_df.columns and len(dashboard_filtered_df) > 0:

                # Normalize casing/whitespace/missing values purely for this
                # chart's grouping (does not modify the underlying data used
                # elsewhere) — otherwise "Yes"/"yes"/"No"/"no" would count as
                # 4 separate categories instead of 2.
                breakdown_series = (
                    dashboard_filtered_df[breakdown_dimension]
                    .fillna("Unknown")
                    .astype(str)
                    .str.strip()
                    .replace("", "Unknown")
                    .str.title()
                )

                breakdown_counts = (
                    breakdown_series
                    .value_counts()
                    .sort_values(ascending=False)
                    .head(8)
                )

                if PLOTLY_AVAILABLE:

                    fig_breakdown = px.bar(
                        x=breakdown_counts.values,
                        y=breakdown_counts.index,
                        orientation="h",
                        labels={"x": "Number of Claims", "y": breakdown_dimension},
                        color=breakdown_counts.values,
                        color_continuous_scale=["#CADCFC", "#1E2761"],
                    )
                    fig_breakdown.update_layout(
                        showlegend=False,
                        coloraxis_showscale=False,
                        margin=dict(t=10, b=10, l=10, r=10),
                        height=320,
                        yaxis=dict(autorange="reversed"),
                    )
                    st.plotly_chart(fig_breakdown, use_container_width=True)

                else:

                    st.bar_chart(breakdown_counts)

            else:

                st.info("No claims match the current filter.")


# ---- Row 2: Scatter (Amount vs Risk) + Histogram (Risk Score Distribution) ----

if show_scatter or show_histogram:

    dash_row2_col1, dash_row2_col2 = st.columns(2)

    if show_scatter:

        with dash_row2_col1:

            st.caption("Claim Amount vs Risk Probability")

            amount_col = next(
                (c for c in ["Claim_Amount", "Total_Claim_Amount", "Amount"]
                 if c in dashboard_filtered_df.columns),
                None,
            )

            if amount_col and len(dashboard_filtered_df) > 0 and PLOTLY_AVAILABLE:

                # Exclude non-positive amounts from the chart only — these
                # are known data-quality errors (e.g. negative claim
                # amounts), not real values, and would distort the axis.
                scatter_df = dashboard_filtered_df[
                    dashboard_filtered_df[amount_col] > 0
                ]
                excluded_count = len(dashboard_filtered_df) - len(scatter_df)

                if excluded_count > 0:
                    st.caption(
                        f"{excluded_count} claim(s) excluded from this chart "
                        f"due to an invalid (zero or negative) {amount_col}."
                    )

                fig_scatter = px.scatter(
                    scatter_df,
                    x=amount_col,
                    y="Risk_Probability",
                    color="Risk_Category",
                    color_discrete_map=RISK_COLORS,
                    hover_data=(
                        ["Claim_ID"] if "Claim_ID" in scatter_df.columns else None
                    ),
                    opacity=0.75,
                )
                fig_scatter.update_layout(
                    margin=dict(t=10, b=10, l=10, r=10),
                    height=320,
                    legend_title_text="",
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

            elif not amount_col:

                st.info("No claim amount column found for this chart.")

            else:

                st.info("No claims match the current filter.")

    if show_histogram:

        with dash_row2_col2:

            st.caption("Risk Score Distribution")

            if len(dashboard_filtered_df) > 0 and PLOTLY_AVAILABLE:

                fig_hist = px.histogram(
                    dashboard_filtered_df,
                    x="Risk_Probability",
                    nbins=20,
                    color_discrete_sequence=["#1E2761"],
                )
                fig_hist.add_vline(x=0.30, line_dash="dash", line_color="#F9A825")
                fig_hist.add_vline(x=0.60, line_dash="dash", line_color="#C62828")
                fig_hist.update_layout(
                    margin=dict(t=10, b=10, l=10, r=10),
                    height=320,
                    xaxis_title="Risk Probability",
                    yaxis_title="Number of Claims",
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            else:

                st.info("No claims match the current filter.")


# ---- Row 3: Top 10 Riskiest Claims + Top Risk Factors (portfolio-wide) ----

if show_leaderboard or show_factors:

    dash_row3_col1, dash_row3_col2 = st.columns(2)

    if show_leaderboard:

        with dash_row3_col1:

            st.caption("Top 10 Riskiest Claims (current filter)")

            if len(dashboard_filtered_df) > 0 and "Claim_ID" in dashboard_filtered_df.columns:

                top10 = (
                    dashboard_filtered_df
                    .sort_values("Risk_Probability", ascending=False)
                    .head(10)
                )

                if PLOTLY_AVAILABLE:

                    fig_top10 = px.bar(
                        top10.sort_values("Risk_Probability"),
                        x="Risk_Probability",
                        y="Claim_ID",
                        orientation="h",
                        color="Risk_Category",
                        color_discrete_map=RISK_COLORS,
                    )
                    fig_top10.update_layout(
                        margin=dict(t=10, b=10, l=10, r=10),
                        height=320,
                        legend_title_text="",
                        xaxis_title="Risk Probability",
                        yaxis_title="",
                    )
                    st.plotly_chart(fig_top10, use_container_width=True)

                else:

                    st.dataframe(
                        top10[["Claim_ID", "Risk_Probability", "Risk_Category"]],
                        use_container_width=True,
                    )

            else:

                st.info("No claims match the current filter.")

    if show_factors:

        with dash_row3_col2:

            st.caption("Top Risk Factors — portfolio-wide, not affected by the filters above")

            try:
                importances = random_forest_model.feature_importances_
                feature_names_all = preprocessor.get_feature_names_out()

                min_len = min(len(importances), len(feature_names_all))

                importance_df = pd.DataFrame({
                    "Feature": [technical_feature_label(f) for f in feature_names_all[:min_len]],
                    "Importance": importances[:min_len],
                })

                importance_df = (
                    importance_df
                    .groupby("Feature", as_index=False)["Importance"]
                    .sum()
                    .sort_values("Importance", ascending=False)
                    .head(6)
                )

                if PLOTLY_AVAILABLE:

                    fig_importance = px.bar(
                        importance_df.sort_values("Importance"),
                        x="Importance",
                        y="Feature",
                        orientation="h",
                        color_discrete_sequence=["#1E2761"],
                    )
                    fig_importance.update_layout(
                        showlegend=False,
                        margin=dict(t=10, b=10, l=10, r=10),
                        height=320,
                        yaxis_title="",
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)

                elif MATPLOTLIB_AVAILABLE:

                    fig_imp2, ax_imp2 = plt.subplots(figsize=(4.2, 2.8))
                    ax_imp2.barh(
                        importance_df["Feature"][::-1],
                        importance_df["Importance"][::-1],
                        color="#1E2761",
                    )
                    st.pyplot(fig_imp2)
                    plt.close(fig_imp2)

                else:

                    st.dataframe(importance_df, use_container_width=True)

            except Exception as e:
                st.info(f"Top risk factors unavailable: {e}")


# ============================================================
# COLOR-CODED RISK TABLE (ALL CLAIMS)
# ============================================================

st.subheader(
    "🎨 All Claims — Color-Coded by Risk"
)

st.caption(
    "Reflects the dashboard filter above."
)


def highlight_risk_row(row):
    """Return a background + text color for the whole row based on Risk_Category.
    Text color is set explicitly (not just background) so rows stay readable
    in both Streamlit's light and dark themes."""
    color = RISK_COLORS.get(row["Risk_Category"], "#FFFFFF")
    # Light-tint background + dark text, so contrast holds regardless of theme
    tint_map = {
        "#2E7D32": "background-color: #E8F5E9; color: #1B1B1B",
        "#F9A825": "background-color: #FFF8E1; color: #1B1B1B",
        "#C62828": "background-color: #FFEBEE; color: #1B1B1B",
    }
    style = tint_map.get(color, "")
    return [style] * len(row)


display_columns = [
    col for col in
    ["Claim_ID", "Risk_Category", "Risk_Probability", "Model_Prediction"]
    if col in final_scored_df.columns
]

st.dataframe(
    dashboard_filtered_df[display_columns].style.apply(highlight_risk_row, axis=1),
    use_container_width=True,
)


# ============================================================
# HIGH-RISK CLAIMS
# ============================================================

st.header(
    "🚨 High-Risk Claims"
)


high_risk_df = final_scored_df[
    final_scored_df[
        "Risk_Category"
    ] == "High"
]


st.write(
    f"High-risk claims: {len(high_risk_df)}"
)


if len(high_risk_df) > 0:

    st.dataframe(
        high_risk_df.style.apply(highlight_risk_row, axis=1),
        use_container_width=True
    )

else:

    st.info(
        "No high-risk claims were detected."
    )


# ============================================================
# FILTER CLAIMS
# ============================================================

st.header(
    "🔎 Filter Scored Claims"
)


risk_filter = st.multiselect(
    "Select Risk Categories",
    options=[
        "Low",
        "Medium",
        "High"
    ],
    default=[
        "Low",
        "Medium",
        "High"
    ]
)


filtered_claims = final_scored_df[
    final_scored_df[
        "Risk_Category"
    ].isin(
        risk_filter
    )
]


st.write(
    f"Filtered claims: {len(filtered_claims)}"
)


st.dataframe(
    filtered_claims.style.apply(highlight_risk_row, axis=1),
    use_container_width=True
)


# ============================================================
# INDIVIDUAL CLAIM
# ============================================================

st.header(
    "👤 Select Individual Claim"
)


if "Claim_ID" in uploaded_df.columns:

    claim_ids = (
        uploaded_df[
            "Claim_ID"
        ]
        .astype(str)
        .tolist()
    )

    selected_claim_id = st.selectbox(
        "Select Claim ID",
        claim_ids
    )

    selected_index = claim_ids.index(
        selected_claim_id
    )

else:

    selected_index = st.selectbox(
        "Select Claim Row",
        range(
            len(uploaded_df)
        )
    )

    selected_claim_id = str(
        selected_index
    )


selected_claim = (
    uploaded_df
    .iloc[
        selected_index
    ]
)


# ============================================================
# SELECTED CLAIM INFORMATION
# ============================================================

st.subheader(
    "📋 Selected Claim Information"
)


st.dataframe(
    selected_claim
    .to_frame(
        name="Value"
    ),
    use_container_width=True
)


# ============================================================
# SELECTED CLAIM RESULT
# ============================================================

selected_prediction = int(
    all_predictions[
        selected_index
    ]
)


selected_probability = float(
    all_probabilities[
        selected_index
    ]
)


selected_risk = get_risk_category(
    selected_probability
)


result_col1, result_col2, result_col3 = (
    st.columns(3)
)


with result_col1:

    st.metric(
        "Fraud Prediction",
        selected_prediction
    )


with result_col2:

    st.metric(
        "Risk Probability",
        f"{selected_probability:.2%}"
    )


with result_col3:

    st.metric(
        "Risk Category",
        selected_risk
    )







# ---- This claim vs. the portfolio average, on its key numeric factors ----

st.caption("This claim compared to the portfolio average")

radar_candidates = [
    "Policy_Tenure_Years", "Previous_Claim_Count", "Vehicle_Age",
    "Submission_Delay_Days", "Invoice_Variance_Percentage",
]
radar_features = [f for f in radar_candidates if f in final_scored_df.columns]

if len(radar_features) >= 3 and PLOTLY_AVAILABLE:

    # Use the 95th percentile, not the raw max, as the scaling reference.
    # A single data-entry error (e.g. an extreme outlier value) would
    # otherwise dominate the scale and squash every normal claim's bar
    # toward zero on that spoke.
    portfolio_scale = (
        final_scored_df[radar_features].astype(float).abs().quantile(0.95).replace(0, 1)
    )

    selected_values_norm = (
        selected_claim[radar_features].astype(float).abs() / portfolio_scale
    ).clip(0, 1.5)

    portfolio_avg_norm = (
        (final_scored_df[radar_features].astype(float).abs().mean()) / portfolio_scale
    ).clip(0, 1.5)

    fig_radar = go.Figure()

    fig_radar.add_trace(go.Scatterpolar(
        r=portfolio_avg_norm.tolist() + [portfolio_avg_norm.tolist()[0]],
        theta=radar_features + [radar_features[0]],
        fill="toself",
        name="Portfolio Average",
        line_color="#9AA3D6",
        opacity=0.5,
    ))

    fig_radar.add_trace(go.Scatterpolar(
        r=selected_values_norm.tolist() + [selected_values_norm.tolist()[0]],
        theta=radar_features + [radar_features[0]],
        fill="toself",
        name=f"Claim {selected_claim_id}",
        line_color=RISK_COLORS.get(selected_risk, "#1E2761"),
    ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1.5])),
        showlegend=True,
        margin=dict(t=30, b=10, l=40, r=40),
        height=380,
    )

    st.plotly_chart(fig_radar, use_container_width=True)

    st.caption(
        "Values are scaled relative to the portfolio's own range, so this "
        "shows how unusual this claim is on each factor, not raw units."
    )

elif len(radar_features) >= 3:

    st.dataframe(
        pd.DataFrame({
            "Factor": radar_features,
            "This Claim": [selected_claim[f] for f in radar_features],
            "Portfolio Average": [final_scored_df[f].mean() for f in radar_features],
        }),
        use_container_width=True,
    )


# ---- Investigator Notes: private notes per claim, kept for this session ----

st.subheader("📝 Investigator Notes")

if "investigator_notes" not in st.session_state:
    st.session_state["investigator_notes"] = {}

existing_note = st.session_state["investigator_notes"].get(selected_claim_id, "")

note_text = st.text_area(
    "Add your notes for this claim",
    value=existing_note,
    placeholder="e.g. Called customer on 2026-09-01, awaiting police report copy...",
    key=f"note_input_{selected_claim_id}",
    height=100,
)

note_col1, note_col2 = st.columns([1, 4])

with note_col1:
    if st.button("💾 Save Note", key=f"save_note_{selected_claim_id}"):
        st.session_state["investigator_notes"][selected_claim_id] = note_text
        st.success("Note saved for this session.")

with note_col2:
    saved_count = len(
        [v for v in st.session_state["investigator_notes"].values() if v.strip()]
    )
    st.caption(
        f"{saved_count} claim(s) have notes saved this session. "
        "Notes are not saved permanently — download the investigation "
        "report below to keep a copy."
    )


# ============================================================
# SHAP CONTRIBUTING FACTORS
# ============================================================

st.header(
    "🔎 Contributing Factors"
)


top_shap = None


try:

    import shap


    # --------------------------------------------------------
    # Get fitted Random Forest
    # --------------------------------------------------------

    fitted_rf_model = random_forest_model


    # --------------------------------------------------------
    # Prepare selected claim
    # --------------------------------------------------------

    selected_model_input = (
        model_input
        .iloc[
            [selected_index]
        ]
        .copy()
    )


    # --------------------------------------------------------
    # Transform selected claim
    # --------------------------------------------------------

    processed_claim = (
        preprocessor
        .transform(
            selected_model_input
        )
    )


    # --------------------------------------------------------
    # SHAP Tree Explainer
    # --------------------------------------------------------

    explainer = shap.TreeExplainer(
        fitted_rf_model
    )


    shap_values = (
        explainer.shap_values(
            processed_claim
        )
    )


    # --------------------------------------------------------
    # SHAP version compatibility
    # --------------------------------------------------------

    if isinstance(
        shap_values,
        list
    ):

        claim_shap_values = (
            np.asarray(
                shap_values[1]
            )[0]
        )

    else:

        shap_array = np.asarray(
            shap_values
        )

        if shap_array.ndim == 3:

            claim_shap_values = (
                shap_array[0, :, 1]
            )

        elif shap_array.ndim == 2:

            claim_shap_values = (
                shap_array[0]
            )

        else:

            claim_shap_values = (
                shap_array
            )


    # --------------------------------------------------------
    # Feature names
    # --------------------------------------------------------

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )


    # Safety check
    min_length = min(
        len(feature_names),
        len(claim_shap_values)
    )


    shap_df = pd.DataFrame({

        "Feature":
            feature_names[
                :min_length
            ],

        "SHAP_Value":
            claim_shap_values[
                :min_length
            ]

    })


    shap_df[
        "Absolute_SHAP"
    ] = (
        shap_df[
            "SHAP_Value"
        ]
        .abs()
    )


    top_shap = (
        shap_df
        .sort_values(
            "Absolute_SHAP",
            ascending=False
        )
        .head(5)
    )


    display_shap = (
        top_shap[
            [
                "Feature",
                "SHAP_Value"
            ]
        ]
        .copy()
    )


    display_shap[
        "Effect"
    ] = np.where(
        display_shap[
            "SHAP_Value"
        ] > 0,
        "Increases risk",
        "Decreases risk"
    )


    st.dataframe(
        display_shap,
        use_container_width=True
    )


except Exception as e:

    st.warning(
        f"SHAP explanation unavailable: {e}"
    )


# ============================================================
# AI-GENERATED EXPLANATION
# ============================================================

st.header(
    "🤖 AI-Generated Explanation"
)


if st.button(
    "Generate AI Explanation"
):

    try:

        gemini_api_key = os.getenv(
            "GEMINI_API_KEY"
        )


        if not gemini_api_key:

            st.warning(
                "GEMINI_API_KEY is not configured. "
                "AI explanation is optional."
            )

        else:

            from google import genai


            client = genai.Client(
                api_key=gemini_api_key
            )


            shap_text = ""


            if top_shap is not None:

                for _, row in top_shap.iterrows():

                    direction = (
                        "increases risk"
                        if row[
                            "SHAP_Value"
                        ] > 0
                        else
                        "decreases risk"
                    )


                    shap_text += (
                        f"- "
                        f"{row['Feature']}: "
                        f"{row['SHAP_Value']:.4f} "
                        f"({direction})\n"
                    )


            prompt = f"""
You are an insurance claim risk analyst.

Explain the following machine-learning prediction
in simple and professional language.

Claim ID:
{selected_claim_id}

Fraud Prediction:
{selected_prediction}

Risk Probability:
{selected_probability:.2%}

Risk Category:
{selected_risk}

Top contributing factors:
{shap_text}

Provide:

1. Short risk summary.
2. Main contributing factors.
3. Recommendation for the claim reviewer.

Do not invent information.
Clearly state that the prediction is an AI-assisted
risk assessment and not a final fraud determination.
"""


            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt
            )


            st.write(
                response.text
            )

            if "ai_explanations" not in st.session_state:
                st.session_state["ai_explanations"] = {}
            st.session_state["ai_explanations"][selected_claim_id] = response.text


    except Exception as e:

        st.error(
            f"AI explanation error: {e}"
        )


# ============================================================
# INVESTIGATION REPORT
# ============================================================

st.header(
    "📄 Investigation Report"
)

st.caption(
    "Combines this claim's score, contributing factors, AI explanation "
    "(if generated above), and your notes into one downloadable report."
)

report_ai_text = st.session_state.get("ai_explanations", {}).get(
    selected_claim_id,
    "(Not generated for this claim yet — click 'Generate AI Explanation' above first if needed.)",
)

report_note_text = st.session_state.get("investigator_notes", {}).get(
    selected_claim_id, ""
).strip()
if not report_note_text:
    report_note_text = "(No investigator notes added for this claim.)"

report_shap_lines = []
if top_shap is not None:
    for _, row in top_shap.iterrows():
        direction = "increases risk" if row["SHAP_Value"] > 0 else "decreases risk"
        report_shap_lines.append(f"- {row['Feature']}: {row['SHAP_Value']:.4f} ({direction})")
report_shap_text = "\n".join(report_shap_lines) if report_shap_lines else "(SHAP factors unavailable.)"

report_text = f"""INSURANCE CLAIM INVESTIGATION REPORT
{'=' * 40}

Claim ID: {selected_claim_id}
Risk Category: {selected_risk}
Risk Probability: {selected_probability:.2%}
Model Prediction: {selected_prediction}

TOP CONTRIBUTING FACTORS
{'-' * 40}
{report_shap_text}

AI-GENERATED EXPLANATION
{'-' * 40}
{report_ai_text}

INVESTIGATOR NOTES
{'-' * 40}
{report_note_text}

{'=' * 40}
This report is AI-assisted and does not represent a final fraud
determination. A human investigator must make the final decision.
"""

st.download_button(
    label="📄 Download Investigation Report",
    data=report_text.encode("utf-8"),
    file_name=f"investigation_report_{selected_claim_id}.txt",
    mime="text/plain",
)


# ============================================================
# DOWNLOAD SCORED CSV
# ============================================================

st.header(
    "📥 Investigator Export — All Scored Claims"
)

st.caption(
    "Full batch export for investigator use. Includes every claim in "
    "this upload, with model predictions and probabilities attached."
)


csv_data = (
    final_scored_df
    .to_csv(
        index=False
    )
    .encode(
        "utf-8"
    )
)


st.download_button(
    label="📥 Download Scored Claims CSV",
    data=csv_data,
    file_name="final_scored_insurance_claims.csv",
    mime="text/csv"
)


# ============================================================
# FINAL DATASET
# ============================================================

st.header(
    "📊 Final Scored Dataset"
)


st.write(
    "Final scored dataset shape:",
    final_scored_df.shape
)


st.dataframe(
    final_scored_df.head(20),
    use_container_width=True
)


# ============================================================
# CUSTOMER VIEW LINK
# ============================================================

st.divider()

st.header(
    "🔗 Customer Claim View"
)

st.write(
    "Share this link with the customer to view "
    "their selected claim."
)

# The app's public address is not something Streamlit reliably knows
# on its own, so it's configured once here instead of hardcoded.
# Locally, this defaults to localhost. On Streamlit Community Cloud,
# set APP_BASE_URL under Settings -> Secrets to your real app URL,
# e.g. APP_BASE_URL = "https://your-app-name.streamlit.app"
try:
    APP_BASE_URL = st.secrets.get("APP_BASE_URL", "http://localhost:8501")
except Exception:
    APP_BASE_URL = "http://localhost:8501"

# Strip any trailing slash(es), no matter how the secret was typed —
# a trailing slash here would otherwise create a double-slash link
# (e.g. ".app//Customer_Claim_View"), which can break Streamlit
# Cloud's page routing entirely.
APP_BASE_URL = APP_BASE_URL.rstrip("/")

customer_link_url = (
    f"{APP_BASE_URL}/"
    f"Customer_Claim_View"
    f"?claim_id={selected_claim_id}"
)

if APP_BASE_URL == "http://localhost:8501":
    st.caption(
        "⚠️ Using a local address. Set `APP_BASE_URL` in this app's "
        "Secrets once deployed, so this link works for real customers."
    )

st.link_button(
    "🙋 Open Customer View",
    customer_link_url
)

st.code(
    customer_link_url,
    language=None
)

st.caption(
    "This link opens only the selected claim."
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Insurance Claim Risk Detection | "
    "Random Forest Machine Learning | "
    "Streamlit"
)
