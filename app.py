import streamlit as st
import pandas as pd
import joblib
import os

# Page
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔐",
    layout="wide"
)

# CSS
st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
}

.subtitle {
    text-align: center;
    color: #888;
    font-size: 18px;
    margin-bottom: 25px;
}

.result-box {
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    margin-top: 25px;
}

.fraud {
    background-color: #ffe5e5;
    border: 2px solid #ff4b4b;
}

.safe {
    background-color: #e5ffe9;
    border: 2px solid #21c354;
}

.result-title {
    font-size: 28px;
    font-weight: bold;
    color: #111;
}

.risk-text {
    font-size: 18px;
    margin-top: 10px;
    color: #222;
}
</style>
""", unsafe_allow_html=True)

# Model
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

# Header
st.markdown(
    '<div class="main-title">🔐 AI Fraud Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning based Transaction Fraud Detection</div>',
    unsafe_allow_html=True
)

st.divider()

# Inputs
st.subheader("💳 Enter Transaction Details")

col1, col2 = st.columns(2)

with col1:
    transaction_type = st.selectbox(
        "Transaction Type",
        ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
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

with col2:
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
        "💡 Enter transaction information and click **Check Transaction**."
    )

st.divider()

# Button
c1, c2, c3 = st.columns([1, 2, 1])

with c2:
    check_transaction = st.button(
        "🔍 Check Transaction",
        use_container_width=True
    )

# Prediction
if check_transaction:

    if amount <= 0:
        st.warning("⚠️ Transaction amount must be greater than 0.")
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
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        fraud_probability = probability[1] * 100

        st.divider()
        st.subheader("📊 Transaction Analysis")

        st.dataframe(
            input_data,
            use_container_width=True,
            hide_index=True
        )

        # Result
        if prediction == 1:
            st.markdown(
                f"""
                <div class="result-box fraud">
                    <div class="result-title">
                        🚨 FRAUDULENT TRANSACTION
                    </div>
                    <div class="risk-text">
                        Fraud Risk Score: <b>{fraud_probability:.2f}%</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.error(
                "⚠️ The AI model has classified this transaction as potentially fraudulent."
            )

        else:
            st.markdown(
                f"""
                <div class="result-box safe">
                    <div class="result-title">
                        ✅ LEGITIMATE TRANSACTION
                    </div>
                    <div class="risk-text">
                        Fraud Risk Score: <b>{fraud_probability:.2f}%</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.success(
                "✅ The AI model has classified this transaction as legitimate."
            )

        # Risk
        st.subheader("🎯 Fraud Probability")

        st.progress(
            min(max(int(fraud_probability), 0), 100)
        )

        if fraud_probability >= 70:
            st.warning("🔴 High fraud risk")
        elif fraud_probability >= 40:
            st.warning("🟠 Medium fraud risk")
        else:
            st.info("🟢 Low fraud risk")

    except Exception as e:
        st.error("❌ Prediction failed.")
        st.exception(e)

st.divider()

st.caption(
    "AI Fraud Detection System | Machine Learning Project"
)