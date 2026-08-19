# 🔐 AI Fraud Detection System

<p align="center">
  <b>Machine Learning powered transaction fraud detection with an interactive Streamlit application.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikit-learn&logoColor=white" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Model-Logistic%20Regression-0B5E75" alt="Model">
  <img src="https://img.shields.io/badge/Status-Completed-22C55E" alt="Status">
</p>

<p align="center">
  <i>An end-to-end Machine Learning project that combines data analysis, feature engineering, imbalanced classification, model deployment, validation, and risk visualization.</i>
</p>

---

## 🚀 Why This Project Stands Out

This project is not just a trained ML model. It demonstrates a complete **ML-to-application workflow**:

**Transaction Data → EDA → Feature Engineering → Preprocessing → Model Training → Evaluation → Model Serialization → Streamlit Deployment → Fraud Risk Prediction**

The application converts raw transaction details into a simple business-facing result:

- **Legitimate / Potentially Fraudulent**
- **Fraud Risk Score**
- **Low / Medium / High Risk**
- **Input validation warnings**

---

## 🎯 Project Objective

Financial fraud detection is a highly imbalanced classification problem where fraudulent transactions are rare compared with legitimate transactions.

The objective of this project is to build a practical Machine Learning system that can:

1. Learn patterns from historical transaction data.
2. Detect potentially fraudulent transactions.
3. Handle class imbalance during model training.
4. Return a fraud probability instead of only a class label.
5. Present the prediction through a user-friendly Streamlit interface.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🤖 ML Classification | Logistic Regression based fraud classifier |
| ⚖️ Imbalance Handling | `class_weight="balanced"` |
| 🔧 Preprocessing Pipeline | Scaling + One-Hot Encoding + classifier |
| 📊 EDA | Fraud distribution, transaction patterns, balance analysis |
| 🎯 Probability Scoring | Fraud probability from `predict_proba()` |
| 🟢🟠🔴 Risk Levels | Low, Medium and High risk interpretation |
| ⚠️ Input Validation | Basic logical checks before prediction |
| 🌐 Streamlit UI | Interactive browser-based application |
| 💾 Model Persistence | Trained pipeline stored using Joblib |
| 🧩 Reusable Pipeline | Same preprocessing used during training and inference |

---

# 🧠 Machine Learning Architecture

```text
                    ┌─────────────────────┐
                    │  Historical Data    │
                    │  6.36M Transactions │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │        EDA          │
                    │ Pattern & Fraud     │
                    │ Analysis            │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Feature Preparation │
                    │ Numerical +         │
                    │ Categorical Data    │
                    └──────────┬──────────┘
                               │
                               ▼
             ┌──────────────────────────────────┐
             │      Scikit-Learn Pipeline       │
             │                                  │
             │  Numerical → StandardScaler      │
             │  Categorical → OneHotEncoder     │
             └────────────────┬─────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │ Logistic Regression │
                    │ class_weight=       │
                    │ "balanced"          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Model Evaluation    │
                    │ Precision / Recall  │
                    │ F1 / Accuracy       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Joblib Model File   │
                    │ .pkl                │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Streamlit App       │
                    │ Real-time Prediction │
                    └─────────────────────┘
```

---

# 📊 Dataset

The project uses a large transaction dataset containing **6,362,620 transactions** and **11 original columns**.

### Dataset Features

| Feature | Description |
|---|---|
| `step` | Transaction time step |
| `type` | Transaction type |
| `amount` | Transaction amount |
| `nameOrig` | Sender identifier |
| `oldbalanceOrg` | Sender balance before transaction |
| `newbalanceOrig` | Sender balance after transaction |
| `nameDest` | Receiver identifier |
| `oldbalanceDest` | Receiver balance before transaction |
| `newbalanceDest` | Receiver balance after transaction |
| `isFraud` | Fraud target variable |
| `isFlaggedFraud` | Existing fraud flag |

### Model Input Features

The deployed model uses:

```text
type
amount
oldbalanceOrg
newbalanceOrig
oldbalanceDest
newbalanceDest
```

Identifier columns and the existing fraud flag are not used as model inputs.

---

# 🔧 Feature Engineering & Preprocessing

The analysis explores balance-based features such as:

```python
balanceDiffOrig = oldbalanceOrg - newbalanceOrig
balanceDiffDest = newbalanceDest - oldbalanceDest
```

The final deployed pipeline applies:

### Numerical Features

- `amount`
- `oldbalanceOrg`
- `newbalanceOrig`
- `oldbalanceDest`
- `newbalanceDest`

**Preprocessing:** `StandardScaler`

### Categorical Feature

- `type`

**Preprocessing:** `OneHotEncoder`

### Classifier

```python
LogisticRegression(
    class_weight="balanced",
    max_iter=1000
)
```

Using balanced class weights helps the classifier pay more attention to the minority fraud class.

---

# 📈 Model Performance

The model was evaluated using a **70/30 stratified train-test split**.

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Legitimate (0) | 1.00 | 0.95 | 0.97 |
| Fraud (1) | 0.02 | **0.94** | 0.04 |
| Overall Accuracy | — | — | **0.95** |

### 🔎 How to Interpret These Results

The model achieves approximately **94% recall for fraud**, meaning it identifies a large proportion of actual fraudulent transactions.

However, fraud precision is low, which means the system can generate false positives.

This is an important real-world fraud detection trade-off:

> **Missing a fraudulent transaction and incorrectly flagging a legitimate transaction have different business costs.**

For a production system, the next step would be threshold optimization and cost-sensitive evaluation rather than relying on accuracy alone.

---

# 🎯 Fraud Risk Scoring

The Streamlit application uses the model's fraud probability to provide an intuitive risk indicator.

| Fraud Probability | Application Result |
|---:|---|
| `< 40%` | 🟢 Low fraud risk |
| `40% – 69%` | 🟠 Medium fraud risk |
| `≥ 70%` | 🔴 High fraud risk |

The application displays both the **prediction** and the **fraud risk score**.

> Note: The model prediction itself is based on the classifier's decision rule; the risk labels are an application-level interpretation of the returned probability.

---

# 🖥️ Application Screenshots

## 1. Transaction Input Interface

The main interface allows the user to enter transaction type, transaction amount, sender balances, and receiver balances.

![Transaction Input Interface](screenshots/app-interface.png)

**What this demonstrates:**

- Clean two-column Streamlit layout
- User-friendly financial inputs
- Transaction type selection
- Validation-ready input form
- Single-click prediction workflow

---

## 2. Legitimate Transaction Result

For a transaction classified as legitimate, the application displays a clear success state with the fraud risk score and a low-risk indicator.

![Legitimate Transaction Result](screenshots/legitimate-transaction.png)

**What this demonstrates:**

- Model prediction
- Fraud probability display
- Positive result visualization
- Low-risk interpretation

---

## 3. Fraudulent Transaction Result

When the model predicts the fraud class, the application highlights the result and displays the estimated fraud risk score.

![Fraudulent Transaction Result](screenshots/fraudulent-transaction.png)

**What this demonstrates:**

- Fraud classification
- Probability-based risk scoring
- Warning-oriented UI
- Visual fraud probability meter
- Medium/high risk interpretation

---

## 4. Input Validation Warning

The application also performs a basic consistency check. For example, it warns the user when the sender's new balance is greater than the old balance.

![Transaction Validation Warning](screenshots/validation-warning.png)

**What this demonstrates:**

- Input validation
- User guidance before/around prediction
- Practical application-level safeguards

---

# 🔄 End-to-End Application Flow

```text
User enters transaction details
              ↓
       Basic validation
              ↓
      Build input DataFrame
              ↓
   Load trained ML pipeline
              ↓
       Predict class
              ↓
   Calculate fraud probability
              ↓
     Assign risk indicator
              ↓
 Display result + probability
```

---

# 📁 Project Structure

```text
Fraud Detection/
│
├── app.py
│   └── Streamlit application
│
├── analysis_model.ipynb
│   └── EDA, feature engineering,
│      model training and evaluation
│
├── AIML Dataset.csv
│   └── Historical transaction dataset
│
├── fraud_detection_pipeline.pkl
│   └── Trained Scikit-Learn pipeline
│
└── screenshots/
    ├── app-interface.png
    ├── legitimate-transaction.png
    ├── fraudulent-transaction.png
    └── validation-warning.png
```

---

# 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core development |
| Pandas | Data manipulation |
| NumPy | Numerical computing |
| Matplotlib | Visualization |
| Seaborn | Statistical visualization |
| Scikit-Learn | Machine Learning |
| Joblib | Model serialization |
| Streamlit | Interactive web application |

---

# 🚀 Installation & Local Setup

## 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Fraud-Detection
```

## 2. Create a virtual environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit joblib
```

## 4. Run the application

```bash
streamlit run app.py
```

The application will open in the browser.

---

# 🧪 Example Prediction

Example input:

```text
Transaction Type: PAYMENT
Transaction Amount: 1000
Sender Balance Before: 5000
Sender Balance After: 4000
Receiver Balance Before: 2000
Receiver Balance After: 3000
```

The application processes these values through the saved ML pipeline and returns:

```text
Prediction
    ↓
Fraud Probability
    ↓
Risk Level
```

The actual prediction depends on the trained model.

---

# 💼 Recruiter / Portfolio Highlights

This project demonstrates practical experience with:

- End-to-end Machine Learning
- Imbalanced binary classification
- Feature preprocessing pipelines
- Model evaluation beyond accuracy
- Probability-based predictions
- Model serialization and reuse
- Streamlit deployment
- Input validation
- Business-oriented risk visualization
- Large-scale transaction data analysis

### Strong Resume Description

> **Developed an end-to-end AI Fraud Detection System using Scikit-Learn and Streamlit, implementing preprocessing pipelines, class-imbalance handling, Logistic Regression, probability-based risk scoring, validation, and an interactive web interface for transaction-level fraud prediction.**

---

# 🔮 Future Improvements

- [ ] Compare Logistic Regression with Random Forest, XGBoost and LightGBM
- [ ] Hyperparameter tuning with cross-validation
- [ ] Precision-Recall curve and ROC-AUC analysis
- [ ] Optimize fraud decision threshold based on business cost
- [ ] SHAP-based explainability
- [ ] Real-time transaction monitoring
- [ ] Transaction history dashboard
- [ ] Database integration
- [ ] FastAPI backend
- [ ] Cloud deployment
- [ ] Authentication and role-based access
- [ ] Automated fraud alerts
- [ ] Model monitoring and drift detection

---

# ⚠️ Limitations

This project is a portfolio/educational implementation.

The current model's low fraud precision indicates that additional work is required before using it in a production financial environment. A production solution would require:

- Cost-sensitive evaluation
- Threshold calibration
- More advanced models
- Robust validation
- Monitoring and drift detection
- Security controls
- Domain-specific testing
- Regulatory and compliance review

---

# 🔐 Disclaimer

This project is intended for **educational and demonstration purposes only**.

It should not be used as a production financial fraud detection system without additional validation, security controls, monitoring, model governance, and domain-specific testing.

---

# 👨‍💻 Author

**Harsh**

**AI Fraud Detection System**  
Machine Learning • Fraud Analytics • Streamlit • Scikit-Learn

---

## ⭐ If You Like This Project

If this project helped you or you found it interesting, consider giving the repository a ⭐ on GitHub.

<p align="center">
  <b>Built with Python • Scikit-Learn • Pandas • Streamlit</b>
</p>
