import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

from heart_model import (
    FEATURES,
    load_framingham_csv,
    predict_one,
    train_from_dataframe
)

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Heart Risk Dashboard", layout="wide")

st.title("❤️ Heart Disease Risk Dashboard")
st.caption("AI-powered 10-year CHD Risk Prediction")

st.divider()

# ================= DATA =================
uploaded = st.file_uploader("📂 Upload Framingham Dataset", type=["csv"])

if uploaded is None:
    st.warning("Upload dataset to continue")
    st.stop()

df = load_framingham_csv(uploaded)
st.success("✅ Dataset Loaded Successfully")

# ================= TRAIN =================
trained = train_from_dataframe(df)

# ================= METRICS =================
st.subheader("📊 Model Performance")

m1, m2, m3 = st.columns(3)
m1.metric("Train Accuracy", f"{trained.train_accuracy:.3f}")
m2.metric("Validation Accuracy", f"{trained.val_accuracy:.3f}")
m3.metric("Test Accuracy", f"{trained.test_accuracy:.3f}")

st.divider()

# ================= VISUALS =================
st.subheader("📊 Model Evaluation")

col1, col2 = st.columns(2)

# ===== ROC =====
with col1:
    st.markdown("### 📈 ROC Curve")

    fpr, tpr, _ = roc_curve(trained.y_test, trained.y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend()

    st.pyplot(fig)

# ===== CONFUSION MATRIX =====
with col2:
    st.markdown("### 🧩 Confusion Matrix")

    fig2, ax2 = plt.subplots(figsize=(4, 4))
    sns.heatmap(
        trained.confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax2
    )

    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")

    st.pyplot(fig2)

st.divider()

# ================= REPORT =================
st.subheader("📄 Classification Report")
st.text(trained.report)

st.divider()

# ================= PREDICTION =================
st.subheader("🔮 Patient Risk Prediction")

with st.form("prediction_form"):

    c1, c2, c3 = st.columns(3)

    inputs = {}

    binary_features = {
        "male", "currentSmoker", "BPMeds",
        "prevalentStroke", "prevalentHyp", "diabetes"
    }

    integer_features = {"age"}

    for i, f in enumerate(FEATURES):

        col = [c1, c2, c3][i % 3]

        # ===== BINARY INPUT =====
        if f in binary_features:
            inputs[f] = col.selectbox(f, [0, 1])

        # ===== INTEGER INPUT =====
        elif f in integer_features:
            inputs[f] = col.number_input(f, min_value=1, max_value=120, step=1, value=45)

        # ===== FLOAT INPUT =====
        else:
            inputs[f] = col.number_input(f, value=0.0)

    submit = st.form_submit_button("Predict Risk")

# ================= RESULT =================
if submit:
    pred, proba = predict_one(trained, inputs)
    risk = proba * 100

    st.subheader("🩺 Risk Assessment")

    r1, r2 = st.columns([1, 2])

    # ===== RISK LABEL =====
    with r1:
        if risk < 30:
            st.success("🟢 LOW RISK")
        elif risk < 70:
            st.warning("🟡 MEDIUM RISK")
        else:
            st.error("🔴 HIGH RISK")

    # ===== DETAILS =====
    with r2:
        st.metric("Risk Probability", f"{risk:.2f}%")
        st.progress(proba)

    st.divider()

    # ===== EXTRA INFO CARD =====
    if risk < 30:
        st.info("✔ Maintain a healthy lifestyle and regular checkups.")
    elif risk < 70:
        st.warning("⚠ Consider lifestyle changes and consult a doctor.")
    else:
        st.error("🚨 High risk detected. Medical consultation recommended immediately.")