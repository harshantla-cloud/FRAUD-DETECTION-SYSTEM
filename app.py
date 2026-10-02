import streamlit as st
import pandas as pd
import joblib
import os

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔐",
    layout="wide"
)

# =========================================================
# SIMPLE CSS
# =========================================================

st.markdown("""
<style>

.main-title {
    text-align: center;
    font-size: 38px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #777;
    font-size: 17px;
    margin-bottom: 25px;
}

.result-box {
    padding: 25px;
    border-radius: 12px;
    text-align: center;
    margin-top: 20px;
}

.fraud {
    background-color: #ffe8e8;
    border: 2px solid #ff4b4b;
}

.safe {
    background-color: #e8f8ed;
    border: 2px solid #21c354;
}

.result-title {
    font-size: 26px;
    font-weight: 700;
}

.risk-text {
    font-size: 18px;
    margin-top: 8px;
}

.footer {
    text-align: center;
    color: #888;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)

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
    st.code("scikit-learn==1.6.1")
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
    '<div class="subtitle">Machine Learning Based Transaction Fraud Detection</div>',
    unsafe_allow_html=True
)

st.divider()

# =========================================================
# INPUT SECTION
# =========================================================

st.subheader("💳 Enter Transaction Details")

col1, col2 = st.columns(2)

# =========================================================
# SENDER DETAILS
# =========================================================

with col1:

    st.markdown("### 📤 Sender Details")

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
# RECEIVER DETAILS
# =========================================================

with col2:

    st.markdown("### 📥 Receiver Details")

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
        "💡 Enter the transaction information and click "
        "**Check Transaction**."
    )

# =========================================================
# BUTTON
# =========================================================

st.divider()

col1, col2, col3 = st.columns([1, 2, 1])

with col2:

    check_transaction = st.button(
        "🔍 Check Transaction",
        use_container_width=True
    )

# =========================================================
# PREDICTION
# =========================================================

if check_transaction:

    if amount <= 0:

        st.warning(
            "⚠️ Transaction amount must be greater than 0."
        )

        st.stop()

    input_data = pd.DataFrame({
        "type": [transaction_type],
        "amount": [amount],
        "oldbalanceOrg": [old_balance_org],
        "newbalanceOrig": [new_balance_orig],
        "oldbalanceDest": [old_balance_dest],
        "newbalanceDest": [new_balance_dest]
    })

    try:

        # Prediction
        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(input_data)[0]

        fraud_probability = probability[1] * 100

        # =================================================
        # ANALYSIS
        # =================================================

        st.divider()

        st.subheader("📊 Transaction Analysis")

        st.dataframe(
            input_data,
            use_container_width=True,
            hide_index=True
        )

        # =================================================
        # RESULT
        # =================================================

        if prediction == 1:

            st.markdown(
                f"""
                <div class="result-box fraud">

                    <div class="result-title">
                        🚨 FRAUDULENT TRANSACTION
                    </div>

                    <div class="risk-text">
                        Fraud Risk Score:
                        <b>{fraud_probability:.2f}%</b>
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.error(
                "⚠️ The AI model has classified this transaction "
                "as potentially fraudulent."
            )

        else:

            st.markdown(
                f"""
                <div class="result-box safe">

                    <div class="result-title">
                        ✅ LEGITIMATE TRANSACTION
                    </div>

                    <div class="risk-text">
                        Fraud Risk Score:
                        <b>{fraud_probability:.2f}%</b>
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.success(
                "✅ The AI model has classified this transaction "
                "as legitimate."
            )

        # =================================================
        # RISK
        # =================================================

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

        if fraud_probability >= 70:

            st.warning("🔴 High fraud risk")

        elif fraud_probability >= 40:

            st.warning("🟠 Medium fraud risk")

        else:

            st.success("🟢 Low fraud risk")

    except Exception as e:

        st.error("❌ Prediction failed.")

        st.exception(e)

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        🔐 AI Fraud Detection System | Machine Learning Project
    </div>
    """,
    unsafe_allow_html=True
)