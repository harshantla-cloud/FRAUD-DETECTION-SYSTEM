import os
import joblib
import pandas as pd
import streamlit as st


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Fraud Detection System",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* Main container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Header */
    .main-title {
        font-size: 44px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #777;
        font-size: 18px;
        margin-bottom: 30px;
    }

    /* Section headers */
    .section-title {
        font-size: 24px;
        font-weight: 700;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    /* Result cards */
    .result-box {
        padding: 28px;
        border-radius: 16px;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    .fraud {
        background: #fff0f0;
        border: 2px solid #ff4b4b;
    }

    .safe {
        background: #effff3;
        border: 2px solid #21c354;
    }

    .result-title {
        font-size: 28px;
        font-weight: 800;
        color: #111;
    }

    .risk-text {
        font-size: 18px;
        margin-top: 10px;
        color: #333;
    }

    /* Info card */
    .info-card {
        padding: 18px;
        border-radius: 12px;
        background: #f7f7f7;
        border: 1px solid #ddd;
        margin-top: 10px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #888;
        font-size: 14px;
        margin-top: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# MODEL CONFIGURATION
# =========================================================

MODEL_PATH = "models/fraud_detection_pipeline.pkl"


@st.cache_resource
def load_model():
    """Load the trained fraud detection pipeline."""
    return joblib.load(MODEL_PATH)


# =========================================================
# MODEL LOADING
# =========================================================

if not os.path.exists(MODEL_PATH):
    st.error(
        "❌ Model file not found. "
        "Please make sure 'models/fraud_detection_pipeline.pkl' exists."
    )
    st.stop()


try:
    model = load_model()

except Exception as e:
    st.error("❌ Unable to load the trained model.")

    st.info(
        "Make sure the deployed environment uses the same "
        "scikit-learn version used during model training."
    )

    st.code("scikit-learn==1.6.1")

    with st.expander("Show technical error"):
        st.exception(e)

    st.stop()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🔐 AI Fraud Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning powered transaction fraud detection'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# =========================================================
# TRANSACTION INPUT
# =========================================================

st.markdown(
    '<div class="section-title">💳 Transaction Details</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)


with col1:

    transaction_type = st.selectbox(
        "Transaction Type",
        options=[
            "PAYMENT",
            "TRANSFER",
            "CASH_OUT",
            "DEBIT",
            "CASH_IN"
        ]
    )

    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=1000.0,
        step=100.0,
        format="%.2f"
    )

    old_balance_org = st.number_input(
        "Sender Balance Before Transaction",
        min_value=0.0,
        value=5000.0,
        step=100.0,
        format="%.2f"
    )

    new_balance_orig = st.number_input(
        "Sender Balance After Transaction",
        min_value=0.0,
        value=4000.0,
        step=100.0,
        format="%.2f"
    )


with col2:

    old_balance_dest = st.number_input(
        "Receiver Balance Before Transaction",
        min_value=0.0,
        value=2000.0,
        step=100.0,
        format="%.2f"
    )

    new_balance_dest = st.number_input(
        "Receiver Balance After Transaction",
        min_value=0.0,
        value=3000.0,
        step=100.0,
        format="%.2f"
    )

    st.markdown(
        """
        <div class="info-card">
        💡 <b>How it works</b><br><br>
        Enter the transaction details and let the trained
        machine learning pipeline classify the transaction
        as legitimate or potentially fraudulent.
        </div>
        """,
        unsafe_allow_html=True
    )


st.divider()


# =========================================================
# PREDICTION BUTTON
# =========================================================

button_col1, button_col2, button_col3 = st.columns([1, 2, 1])

with button_col2:

    check_transaction = st.button(
        "🔍 Analyze Transaction",
        use_container_width=True,
        type="primary"
    )


# =========================================================
# PREDICTION
# =========================================================

if check_transaction:

    # -----------------------------------------------------
    # INPUT VALIDATION
    # -----------------------------------------------------

    if amount <= 0:

        st.warning(
            "⚠️ Transaction amount must be greater than 0."
        )

        st.stop()


    # -----------------------------------------------------
    # CREATE INPUT DATAFRAME
    # -----------------------------------------------------

    input_data = pd.DataFrame(
        {
            "type": [transaction_type],
            "amount": [amount],
            "oldbalanceOrg": [old_balance_org],
            "newbalanceOrig": [new_balance_orig],
            "oldbalanceDest": [old_balance_dest],
            "newbalanceDest": [new_balance_dest]
        }
    )


    # -----------------------------------------------------
    # MODEL PREDICTION
    # -----------------------------------------------------

    try:

        prediction = model.predict(input_data)[0]

        probabilities = model.predict_proba(input_data)[0]

        fraud_probability = probabilities[1] * 100


        # -------------------------------------------------
        # ANALYSIS SECTION
        # -------------------------------------------------

        st.divider()

        st.markdown(
            '<div class="section-title">📊 Transaction Analysis</div>',
            unsafe_allow_html=True
        )

        st.dataframe(
            input_data,
            use_container_width=True,
            hide_index=True
        )


        # -------------------------------------------------
        # FRAUD RESULT
        # -------------------------------------------------

        if prediction == 1:

            st.markdown(
                f"""
                <div class="result-box fraud">

                    <div class="result-title">
                        🚨 FRAUDULENT TRANSACTION
                    </div>

                    <div class="risk-text">
                        Fraud Probability:
                        <b>{fraud_probability:.2f}%</b>
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.error(
                "The machine learning model has classified "
                "this transaction as potentially fraudulent."
            )


        # -------------------------------------------------
        # LEGITIMATE RESULT
        # -------------------------------------------------

        else:

            st.markdown(
                f"""
                <div class="result-box safe">

                    <div class="result-title">
                        ✅ LEGITIMATE TRANSACTION
                    </div>

                    <div class="risk-text">
                        Fraud Probability:
                        <b>{fraud_probability:.2f}%</b>
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.success(
                "The machine learning model has classified "
                "this transaction as legitimate."
            )


        # -------------------------------------------------
        # RISK INDICATOR
        # -------------------------------------------------

        st.markdown(
            '<div class="section-title">🎯 Fraud Risk Indicator</div>',
            unsafe_allow_html=True
        )

        st.progress(
            min(max(int(fraud_probability), 0), 100)
        )


        if fraud_probability >= 70:

            st.warning(
                "🔴 High Fraud Risk — The transaction requires "
                "careful review."
            )

        elif fraud_probability >= 40:

            st.warning(
                "🟠 Medium Fraud Risk — The transaction "
                "shows some risk indicators."
            )

        else:

            st.info(
                "🟢 Low Fraud Risk — The model predicts a "
                "lower probability of fraud."
            )


        # -------------------------------------------------
        # MODEL OUTPUT
        # -------------------------------------------------

        with st.expander("🔎 View Model Output"):

            st.write(
                {
                    "Prediction": int(prediction),
                    "Fraud Probability": f"{fraud_probability:.2f}%",
                    "Transaction Type": transaction_type,
                    "Transaction Amount": amount
                }
            )


    except Exception as e:

        st.error(
            "❌ Prediction failed. Please verify the input "
            "data and model configuration."
        )

        with st.expander("Show technical error"):

            st.exception(e)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        🔐 AI Fraud Detection System
        &nbsp;|&nbsp;
        Machine Learning Project
    </div>
    """,
    unsafe_allow_html=True
)

