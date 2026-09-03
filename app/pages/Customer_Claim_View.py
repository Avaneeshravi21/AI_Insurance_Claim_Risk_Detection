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


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Your Claim Status",
    page_icon="🙋",
)


# ============================================================
# PATHS
# ============================================================
# This file lives one folder deeper than app.py (inside "pages/"),
# so it needs one extra dirname() call to reach the same project root.

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "random_forest_pipeline.pkl"
)

SHARED_DATA_PATH = os.path.join(
    BASE_DIR,
    "shared_data",
    "latest_scored_claims.csv"
)


# ============================================================
# REQUIRED MODEL FEATURES
# (must exactly match app.py's list, since both must agree on
#  what a fully-engineered claim row looks like)
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
# LOAD MODEL (same loading logic as app.py, kept identical
# on purpose so both pages always agree on the same model)
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def get_risk_category(probability):
    if probability >= 0.60:
        return "High"
    elif probability >= 0.30:
        return "Medium"
    else:
        return "Low"


def draw_risk_gauge(probability, title="Risk Level"):
    if not MATPLOTLIB_AVAILABLE:
        return None

    probability = max(0.0, min(1.0, float(probability)))
    fig, ax = plt.subplots(figsize=(4, 2.6), subplot_kw={"aspect": "equal"})

    zones = [
        (0.0, 0.30, "#2E7D32"),
        (0.30, 0.60, "#F9A825"),
        (0.60, 1.0, "#C62828"),
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
    cleaned = raw_feature_name
    if "__" in cleaned:
        cleaned = cleaned.split("__", 1)[1]
    if cleaned in FRIENDLY_FACTOR_MAP:
        return FRIENDLY_FACTOR_MAP[cleaned]
    for base_name, phrase in FRIENDLY_FACTOR_MAP.items():
        if cleaned.startswith(base_name):
            return phrase
    return cleaned.replace("_", " ").lower()


# ============================================================
# LOAD SHARED CLAIMS DATA + MODEL
# ============================================================

st.title("🙋 Your Claim Status")

query_claim_id = st.query_params.get("claim_id", None)

if not query_claim_id:
    st.info(
        "No claim was specified in this link. Please use the link "
        "sent to you, which includes your claim ID."
    )
    st.stop()

try:
    combined_df = pd.read_csv(SHARED_DATA_PATH)

except Exception as e:
    st.error(f"Error loading claims data: {e}")
    st.write("Expected file path:", SHARED_DATA_PATH)
    st.write("File exists:", os.path.exists(SHARED_DATA_PATH))
    st.stop()

if "Claim_ID" not in combined_df.columns:
    st.error("Claims data is missing an expected column. Please contact support.")
    st.stop()

matching_rows = combined_df[combined_df["Claim_ID"].astype(str) == str(query_claim_id)]

if len(matching_rows) == 0:
    st.warning(
        f"We couldn't find a claim matching ID **{query_claim_id}**. "
        "Please check your link, or contact support for help."
    )
    st.stop()

selected_claim = matching_rows.iloc[0]
selected_claim_id = str(selected_claim["Claim_ID"])
selected_risk = selected_claim.get("Risk_Category", "Medium")
selected_probability = float(selected_claim.get("Risk_Probability", 0.5))

try:
    loaded_pipeline = load_model()
    random_forest_pipeline = None
    random_forest_model = None
    preprocessor = None

    if isinstance(loaded_pipeline, dict):
        preprocessor = loaded_pipeline.get("preprocessor")
        random_forest_model = loaded_pipeline.get("model")
    else:
        random_forest_pipeline = loaded_pipeline
        try:
            preprocessor = random_forest_pipeline["preprocessor"]
        except Exception:
            preprocessor = None
        try:
            random_forest_model = random_forest_pipeline["model"]
        except Exception:
            random_forest_model = loaded_pipeline
except Exception:
    preprocessor = None
    random_forest_model = None


# ---- Compute this claim's top contributing factors (for "What we looked at") ----

top_shap = None

try:
    import shap

    missing_features = [f"MF__{f}" for f in MODEL_FEATURES if f"MF__{f}" not in combined_df.columns]

    if preprocessor is not None and random_forest_model is not None and not missing_features:

        model_feature_columns = [f"MF__{f}" for f in MODEL_FEATURES]
        selected_model_input = matching_rows.iloc[[0]][model_feature_columns].copy()
        selected_model_input.columns = MODEL_FEATURES  # strip prefix back for the preprocessor

        processed_claim = preprocessor.transform(selected_model_input)

        explainer = shap.TreeExplainer(random_forest_model)
        shap_values = explainer.shap_values(processed_claim)

        if isinstance(shap_values, list):
            claim_shap_values = np.asarray(shap_values[1])[0]
        else:
            shap_array = np.asarray(shap_values)
            if shap_array.ndim == 3:
                claim_shap_values = shap_array[0, :, 1]
            elif shap_array.ndim == 2:
                claim_shap_values = shap_array[0]
            else:
                claim_shap_values = shap_array

        feature_names = preprocessor.get_feature_names_out()
        min_length = min(len(feature_names), len(claim_shap_values))

        shap_df = pd.DataFrame({
            "Feature": feature_names[:min_length],
            "SHAP_Value": claim_shap_values[:min_length],
        })
        shap_df["Absolute_SHAP"] = shap_df["SHAP_Value"].abs()
        top_shap = shap_df.sort_values("Absolute_SHAP", ascending=False).head(5)

except Exception:
    top_shap = None


# ============================================================
# CUSTOMER-FRIENDLY VIEW
# (identical content/format to the Claim Explanation section
#  in app.py — nothing here has been reworded or restyled)
# ============================================================

st.caption(
    "Written for the customer, not the investigator. No technical "
    "scores, probabilities, or model terminology are shown here."
)

st.subheader("📋 Claim Information")

CUSTOMER_HIDDEN_COLUMNS = [
    "Risk_Category", "Risk_Probability", "Model_Prediction",
    "Submission_Delay_Days", "Invoice_Variance_Percentage",
    "Missing_Information_Count", "Claim_Repair_Ratio",
    "Claims_Per_Tenure_Year", "High_Value_Claim", "Tenure_Group",
    "Description_Length", "Description_Word_Count",
]

customer_visible_fields = [
    col for col in selected_claim.index
    if col not in CUSTOMER_HIDDEN_COLUMNS and not col.startswith("MF__")
]

st.dataframe(
    selected_claim[customer_visible_fields].to_frame().T,
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "This is everything we have on file for your claim — no scores "
    "or internal model details, just your claim's own information."
)

st.divider()

CUSTOMER_STATUS_MESSAGES = {
    "Low": {
        "badge": "✅ Looking Good",
        "message": (
            "Your claim looks straightforward and is progressing "
            "smoothly through our standard process."
        ),
    },
    "Medium": {
        "badge": "🔍 Standard Review",
        "message": (
            "Your claim is going through a bit of extra standard "
            "review. This is routine and does not mean there is "
            "anything wrong with your claim."
        ),
    },
    "High": {
        "badge": "🕵️ Detailed Review",
        "message": (
            "Your claim has been selected for a more detailed review "
            "by our team, as part of our normal process for certain "
            "types of claims."
        ),
    },
}

status_info = CUSTOMER_STATUS_MESSAGES.get(
    selected_risk,
    CUSTOMER_STATUS_MESSAGES["Medium"],
)

customer_col1, customer_col2 = st.columns([1, 1.4])

with customer_col1:

    st.subheader(status_info["badge"])
    st.write(status_info["message"])

    if MATPLOTLIB_AVAILABLE:
        customer_gauge_fig = draw_risk_gauge(
            selected_probability, "Claim Status"
        )
        for text_obj in customer_gauge_fig.axes[0].texts:
            if "%" in text_obj.get_text():
                text_obj.set_text("")
        st.pyplot(customer_gauge_fig)
        plt.close(customer_gauge_fig)

with customer_col2:

    st.markdown("**What we looked at:**")

    shown_phrases = set()

    if top_shap is not None and len(top_shap) > 0:

        shown = 0

        for _, row in top_shap.iterrows():

            if shown >= 3:
                break

            friendly_phrase = humanize_feature_name(row["Feature"])

            if friendly_phrase in shown_phrases:
                continue

            st.write(f"• {friendly_phrase.capitalize()}")
            shown_phrases.add(friendly_phrase)
            shown += 1

    else:
        st.write(
            "• Your policy and claim details were reviewed as part "
            "of our standard process."
        )

    st.markdown("**What happens next:**")

    stages = ["Claim Received", "Under Review", "Decision Made"]
    current_stage_index = {"Low": 2, "Medium": 1, "High": 1}.get(
        selected_risk, 1
    )

    stage_display = " → ".join(
        f"**[{s}]**" if i == current_stage_index else s
        for i, s in enumerate(stages)
    )
    st.write(stage_display)

    st.caption(
        "You do not need to take any action right now. If we need "
        "more information from you, our team will reach out directly."
    )

    customer_summary_lines = [
        f"Claim ID: {selected_claim_id}",
        f"Status: {status_info['badge']}",
        f"Summary: {status_info['message']}",
        "",
        "What we looked at:",
    ]
    for phrase in shown_phrases:
        customer_summary_lines.append(f"- {phrase.capitalize()}")

    customer_summary_text = "\n".join(customer_summary_lines).encode("utf-8")

    st.download_button(
        label="📥 Download My Claim Summary",
        data=customer_summary_text,
        file_name=f"claim_summary_{selected_claim_id}.txt",
        mime="text/plain",
        help="Contains only this claim's status — not raw scores, "
             "and not any other claim's information.",
    )

    st.divider()

    st.subheader("📋 Your Claim at a Glance")

    glance_fields = [
        ("Claim ID", "Claim_ID"),
        ("Incident Type", "Incident_Type"),
        ("Incident Date", "Incident_Date"),
        ("Claim Amount", "Claim_Amount"),
        ("Policy Type", "Policy_Type"),
    ]

    glance_col1, glance_col2 = st.columns(2)
    glance_items = [
        (label, selected_claim[col])
        for label, col in glance_fields
        if col in combined_df.columns
    ]

    for i, (label, value) in enumerate(glance_items):
        target_col = glance_col1 if i % 2 == 0 else glance_col2
        with target_col:
            if label == "Claim Amount":
                try:
                    st.metric(label, f"${float(value):,.2f}")
                except (ValueError, TypeError):
                    st.metric(label, str(value))
            else:
                st.metric(label, str(value))

    st.caption(
        "This is exactly what we have on file for your claim. If "
        "anything here looks incorrect, please contact us below."
    )

    st.divider()

    st.subheader("📶 Claim Progress")

    progress_fraction = {"Low": 0.9, "Medium": 0.6, "High": 0.6}.get(
        selected_risk, 0.5
    )
    st.progress(
        progress_fraction,
        text=f"{int(progress_fraction * 100)}% through our standard process",
    )

    st.divider()

    st.subheader("❓ Common Questions")

    with st.expander("Why does my claim need additional review?"):
        st.write(
            "Some claims are routinely selected for a closer look as "
            "part of our standard process. This is normal and does "
            "not mean there is a problem with your claim."
        )

    with st.expander("Will this affect my premium?"):
        st.write(
            "A claim being reviewed does not, by itself, affect your "
            "premium. Any changes to your policy would be communicated "
            "to you separately and clearly."
        )

    with st.expander("How long will this take?"):
        st.write(
            "Most reviews are completed within our standard "
            "processing time. If we need anything further from you, "
            "our team will reach out directly with next steps."
        )

    with st.expander("What should I do right now?"):
        st.write(
            "Nothing — no action is needed on your part unless we "
            "contact you directly to request more information."
        )

    st.divider()

    st.subheader("💬 Was this explanation helpful?")

    feedback_col1, feedback_col2, feedback_col3 = st.columns([1, 1, 4])

    if "claim_feedback" not in st.session_state:
        st.session_state["claim_feedback"] = {}

    with feedback_col1:
        if st.button("👍 Yes", key=f"feedback_yes_{selected_claim_id}"):
            st.session_state["claim_feedback"][selected_claim_id] = "helpful"

    with feedback_col2:
        if st.button("👎 No", key=f"feedback_no_{selected_claim_id}"):
            st.session_state["claim_feedback"][selected_claim_id] = "not_helpful"

    with feedback_col3:
        current_feedback = st.session_state["claim_feedback"].get(selected_claim_id)
        if current_feedback == "helpful":
            st.caption("Thanks for letting us know!")
        elif current_feedback == "not_helpful":
            st.caption("Thanks — we'll use this to improve our explanations.")

    st.divider()

    st.subheader("📞 Need Help?")
    st.info(
        "**Customer Support**\n\n"
        "📧 support@example-insurance.com\n\n"
        "📞 1-800-555-0100 (Mon–Fri, 9am–6pm)\n\n"
        "Have your Claim ID ready when you contact us: "
        f"**{selected_claim_id}**"
    )