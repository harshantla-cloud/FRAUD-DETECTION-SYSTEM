import streamlit as st
import pandas as pd
import joblib
import os

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Fraud Detection",
    page_icon="🔐",
    layout="wide"
)

# =========================================================
# MODEL
# =========================================================

MODEL_PATH = "models/fraud_detection_pipeline.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found!")
    st.stop()

try:
    model = load_model()

except Exception as e:
    st.error("❌ Model could not be loaded.")
    st.info("Required scikit-learn version: 1.6.1")
    st.exception(e)
    st.stop()

# =========================================================
# HEADER
# =========================================================

st.title("🔐 AI Fraud Detection System")

st.write(
    "Machine Learning based transaction fraud detection "
    "and risk analysis."
)

st.divider()

# =========================================================
# PROJECT INFO
# =========================================================

info1, info2, info3, info4 = st.columns(4)

with info1:
    st.metric("Technology", "Machine Learning")

with info2:
    st.metric("Prediction", "Real-Time")

with info3:
    st.metric("Output", "Fraud / Safe")

with info4:
    st.metric("Analysis", "Risk Score")

st.divider()

# =========================================================
# INPUT SECTION
# =========================================================

st.header("💳 Transaction Details")

st.write(
    "Enter the transaction information below and "
    "click **Analyze Transaction**."
)

col1, col2 = st.columns(2)

# =========================================================
# SENDER
# =========================================================

with col1:

    st.subheader("📤 Sender Information")

    transaction_type = st.selectbox(
        "Transaction Type",
        [
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
        step=100.0
    )

    old_balance_org = st.number_input(
        "Sender Balance Before Transaction",
        min_value=0.0,
        value=5000.0,
        step=100.0
    )

    new_balance_orig = st.number_input(
        "Sender Balance After Transaction",
        min_value=0.0,
        value=4000.0,
        step=100.0
    )

# =========================================================
# RECEIVER
# =========================================================

with col2:

    st.subheader("📥 Receiver Information")

    old_balance_dest = st.number_input(
        "Receiver Balance Before Transaction",
        min_value=0.0,
        value=2000.0,
        step=100.0
    )

    new_balance_dest = st.number_input(
        "Receiver Balance After Transaction",
        min_value=0.0,
        value=3000.0,
        step=100.0
    )

    st.info(
        "💡 The model analyzes transaction type, amount "
        "and sender/receiver balances."
    )

# =========================================================
# BUTTON
# =========================================================

st.divider()

button_col1, button_col2, button_col3 = st.columns([1, 2, 1])

with button_col2:

    check_transaction = st.button(
        "🔍 Analyze Transaction",
        use_container_width=True
    )

# =========================================================
# PREDICTION
# =========================================================

if check_transaction:

    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    if amount <= 0:

        st.warning(
            "⚠️ Transaction amount must be greater than 0."
        )

        st.stop()

    # -----------------------------------------------------
    # INPUT DATA
    # -----------------------------------------------------

    input_data = pd.DataFrame({
        "type": [transaction_type],
        "amount": [amount],
        "oldbalanceOrg": [old_balance_org],
        "newbalanceOrig": [new_balance_orig],
        "oldbalanceDest": [old_balance_dest],
        "newbalanceDest": [new_balance_dest]
    })

    try:

        # -------------------------------------------------
        # PREDICTION
        # -------------------------------------------------

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(input_data)[0]

        fraud_probability = probability[1] * 100

        # -------------------------------------------------
        # ANALYSIS
        # -------------------------------------------------

        st.divider()

        st.header("📊 Transaction Analysis")

        st.dataframe(
            input_data,
            use_container_width=True,
            hide_index=True
        )

        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        st.subheader("🧠 Prediction Result")

        if prediction == 1:

            st.error(
                "🚨 FRAUDULENT TRANSACTION"
            )

            st.metric(
                "Fraud Risk Score",
                f"{fraud_probability:.2f}%"
            )

            st.warning(
                "The machine learning model has classified "
                "this transaction as potentially fraudulent."
            )

        else:

            st.success(
                "✅ LEGITIMATE TRANSACTION"
            )

            st.metric(
                "Fraud Risk Score",
                f"{fraud_probability:.2f}%"
            )

            st.success(
                "The machine learning model has classified "
                "this transaction as legitimate."
            )

        # -------------------------------------------------
        # RISK SCORE
        # -------------------------------------------------

        st.subheader("🎯 Fraud Probability")

        st.progress(
            min(
                max(
                    int(fraud_probability),
                    0
                ),
                100
            )
        )

        st.write(
            f"Estimated fraud probability: "
            f"**{fraud_probability:.2f}%**"
        )

        # -------------------------------------------------
        # RISK LEVEL
        # -------------------------------------------------

        st.subheader("📌 Risk Level")

        if fraud_probability >= 70:

            st.error(
                "🔴 HIGH RISK"
            )

            st.write(
                "The transaction has a high estimated "
                "fraud probability according to the model."
            )

        elif fraud_probability >= 40:

            st.warning(
                "🟠 MEDIUM RISK"
            )

            st.write(
                "The transaction falls within the medium "
                "estimated fraud-risk range."
            )

        else:

            st.success(
                "🟢 LOW RISK"
            )

            st.write(
                "The transaction has a low estimated "
                "fraud probability according to the model."
            )

    except Exception as e:

        st.error("❌ Prediction failed.")

        st.exception(e)

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🔐 AI Fraud Detection System | "
    "Machine Learning Project | Transaction Risk Analysis"
)

