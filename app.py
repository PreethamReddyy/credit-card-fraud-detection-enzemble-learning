import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import joblib

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

html, body, [class*="css"]  {
    font-family: 'Segoe UI', sans-serif;
}

.main {
    background: linear-gradient(to right, #0f172a, #111827);
    color: white;
}

.title {
    font-size: 50px;
    font-weight: 700;
    color: white;
    text-align: center;
    margin-bottom: 10px;
}

.subtitle {
    font-size: 18px;
    text-align: center;
    color: #cbd5e1;
    margin-bottom: 30px;
}

.card {
    background-color: #1e293b;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.4);
    text-align: center;
}

.metric-title {
    color: #94a3b8;
    font-size: 16px;
}

.metric-value {
    color: white;
    font-size: 32px;
    font-weight: bold;
}

.prediction-normal {
    background-color: #14532d;
    color: white;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    font-size: 26px;
    font-weight: bold;
}

.prediction-fraud {
    background-color: #7f1d1d;
    color: white;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    font-size: 26px;
    font-weight: bold;
}

.stButton>button {
    width: 100%;
    background-color: #2563eb;
    color: white;
    font-size: 18px;
    font-weight: bold;
    border-radius: 12px;
    height: 55px;
    border: none;
}

.stButton>button:hover {
    background-color: #1d4ed8;
    color: white;
}

.footer {
    text-align: center;
    color: #94a3b8;
    margin-top: 40px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# MODEL CLASS
# =========================================================
class FraudANN(nn.Module):

    def __init__(self, input_size):
        super(FraudANN, self).__init__()

        self.net = nn.Sequential(

            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 32),
            nn.ReLU(),

            nn.Linear(32, 32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(16, 16),
            nn.ReLU(),

            nn.Linear(16, 16),
            nn.ReLU(),

            nn.Linear(16, 8),
            nn.ReLU(),

            nn.Linear(8, 8),
            nn.ReLU(),

            nn.Linear(8, 8),
            nn.ReLU(),

            nn.Linear(8, 4),
            nn.ReLU(),

            nn.Linear(4, 4),
            nn.ReLU(),

            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# =========================================================
# LOAD MODEL & SCALER
# =========================================================
model = FraudANN(30)
model.load_state_dict(torch.load("fraud_model.pth"))
model.eval()

scaler = joblib.load("scaler.pkl")

# =========================================================
# HEADER
# =========================================================
st.markdown('<div class="title">💳 Credit Card Fraud Detection</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">AI Powered Fraud Transaction Detection System using ANN, SVM and KNN</div>',
    unsafe_allow_html=True
)

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("📌 About Project")

st.sidebar.info("""
### Algorithms Used
✅ ANN  
✅ SVM  
✅ KNN  

### Techniques Used
✅ SMOTE Balancing  
✅ StandardScaler  
✅ Deep Learning  

### Dataset
European Credit Card Fraud Dataset
""")

st.sidebar.success("Best Performing Model: ANN")

# =========================================================
# METRICS
# =========================================================
st.subheader("📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="card">
        <div class="metric-title">Accuracy</div>
        <div class="metric-value">99.86%</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <div class="metric-title">Precision</div>
        <div class="metric-value">58.04%</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
        <div class="metric-title">Recall</div>
        <div class="metric-value">84.69%</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="card">
        <div class="metric-title">ROC-AUC</div>
        <div class="metric-value">98.32%</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.write("")

# =========================================================
# MODEL SELECTION
# =========================================================
st.subheader("🧠 Select Prediction Model")

selected_model = st.selectbox(
    "Choose Model",
    ["ANN", "SVM", "KNN"]
)

st.write(f"Selected Model: **{selected_model}**")

# =========================================================
# INPUT SECTION
# =========================================================
st.subheader("🧾 Enter Transaction Features")

st.info("Enter exactly 30 feature values separated using commas.")

sample_data = """1.60760000e+05,-6.74466065e-01,1.40810502e+00,-1.11062205e+00,
-1.32836578e+00,1.38899603e+00,-1.30843907e+00,1.88587890e+00,
-6.14232966e-01,3.11652212e-01,6.50757004e-01,-8.57784662e-01,
-2.29961446e-01,-1.99817005e-01,2.66371326e-01,-4.65441685e-02,
-7.41398090e-01,-6.05616644e-01,-3.92568188e-01,-1.62648311e-01,
3.94321821e-01,8.00842396e-02,8.10033596e-01,-2.24327230e-01,
7.07899237e-01,-1.35837023e-01,4.51021965e-02,5.33837219e-01,
2.91319253e-01,2.30000000e+01"""

input_text = st.text_area(
    "Transaction Input",
    sample_data,
    height=180
)

# =========================================================
# PREDICTION FUNCTION
# =========================================================
def predict_transaction(data):

    input_data = np.array(data).reshape(1, -1)

    input_scaled = scaler.transform(input_data)

    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

    with torch.no_grad():
        output = model(input_tensor)

        probability = output.item()

        prediction = 1 if probability >= 0.5 else 0

    return prediction, probability

# =========================================================
# PREDICTION BUTTON
# =========================================================
if st.button("🚀 Predict Transaction"):

    try:

        values = [float(x.strip()) for x in input_text.replace("\n", "").split(",")]

        if len(values) != 30:

            st.error("❌ Please enter exactly 30 values.")

        else:

            prediction, probability = predict_transaction(values)

            st.write("")

            if prediction == 1:

                st.markdown(f"""
                <div class="prediction-fraud">
                🚨 Fraudulent Transaction Detected<br><br>
                Fraud Probability: {probability:.4f}
                </div>
                """, unsafe_allow_html=True)

            else:

                st.markdown(f"""
                <div class="prediction-normal">
                ✅ Legitimate Transaction<br><br>
                Safe Probability: {(1 - probability):.4f}
                </div>
                """, unsafe_allow_html=True)

    except:

        st.error("❌ Invalid input format.")

# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div class="footer">
Developed by GROUP 1138 | Cyber Project
</div>
""", unsafe_allow_html=True)