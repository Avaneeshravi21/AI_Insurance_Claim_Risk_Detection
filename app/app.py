import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

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
            errors="coerce"
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
            errors="coerce"
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


with risk_col1:

    st.metric(
        "🟢 Low Risk",
        low_count
    )


with risk_col2:

    st.metric(
        "🟡 Medium Risk",
        medium_count
    )


with risk_col3:

    st.metric(
        "🔴 High Risk",
        high_count
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
        high_risk_df,
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
    filtered_claims,
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
                model="gemini-2.5-flash",
                contents=prompt
            )


            st.write(
                response.text
            )


    except Exception as e:

        st.error(
            f"AI explanation error: {e}"
        )


# ============================================================
# DOWNLOAD SCORED CSV
# ============================================================

st.header(
    "📥 Download Scored Claims"
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
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Insurance Claim Risk Detection | "
    "Random Forest Machine Learning | "
    "Streamlit"
)

