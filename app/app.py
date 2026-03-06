import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
@st.cache_data
def load_data():
    url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
    return pd.read_csv(url)

data = load_data()
# ---------------------------------------------------
# Load Data
# ---------------------------------------------------

@st.cache_data
def load_data():
    url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
    return pd.read_csv(url)
# ---------------------------------------------------
# Load Model
# ---------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("models/fraud_model.pkl")

model = load_model()

# ---------------------------------------------------
# Title
# ---------------------------------------------------

st.title("💳 Credit Card Fraud Detection Dashboard")

st.markdown(
"""
This dashboard analyzes credit card transactions and detects fraud patterns
using machine learning.
"""
)

st.divider()

# ---------------------------------------------------
# Top Navigation Tabs
# ---------------------------------------------------

tab1, tab2, tab3 = st.tabs([
    "Overview",
    "Fraud Analytics",
    "Model Performance"
])

# ---------------------------------------------------
# OVERVIEW TAB
# ---------------------------------------------------

with tab1:

    st.subheader("Dataset Overview")

    total_transactions = len(data)
    fraud_transactions = data["Class"].sum()
    fraud_rate = fraud_transactions / total_transactions * 100

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Transactions", f"{total_transactions:,}")
    col2.metric("Fraud Transactions", f"{fraud_transactions:,}")
    col3.metric("Fraud Rate", f"{fraud_rate:.3f}%")

    st.divider()

    st.subheader("Transaction Amount Distribution")

    fig, ax = plt.subplots()
    sns.histplot(data["Amount"], bins=50, ax=ax)

    ax.set_xlabel("Transaction Amount")
    ax.set_ylabel("Count")

    st.pyplot(fig)

# ---------------------------------------------------
# FRAUD ANALYTICS TAB
# ---------------------------------------------------

st.subheader("Filter Transactions")

amount_range = st.slider(
    "Filter by Amount",
    0,
    int(data["Amount"].max()),
    (0, 1000)
)

filtered_data = data[
    (data["Amount"] >= amount_range[0]) &
    (data["Amount"] <= amount_range[1])
]

st.write("Filtered Transactions:", len(filtered_data))
with tab2:

    st.subheader("Fraud vs Normal Transactions")

    fraud_counts = data["Class"].value_counts()

    fig, ax = plt.subplots()

    ax.pie(
        fraud_counts,
        labels=["Normal", "Fraud"],
        autopct="%1.2f%%",
        colors=["#4CAF50", "#FF4B4B"]
    )

    st.pyplot(fig)

    st.divider()

    st.subheader("Transaction Amount by Class")

    fig, ax = plt.subplots()

    sns.boxplot(
        x=data["Class"],
        y=data["Amount"],
        ax=ax
    )

    ax.set_xticklabels(["Normal", "Fraud"])

    st.pyplot(fig)
    st.divider()
st.subheader("Fraud Activity Over Time")

fig, ax = plt.subplots()

fraud_data = data[data["Class"] == 1]

sns.histplot(fraud_data["Time"], bins=50, ax=ax)

ax.set_xlabel("Transaction Time")
ax.set_ylabel("Fraud Count")

st.pyplot(fig)

# ---------------------------------------------------
# MODEL PERFORMANCE TAB
# ---------------------------------------------------

with tab3:

    st.subheader("Model Performance")

    X = data.drop("Class", axis=1)
    y = data["Class"]

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots()

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Fraud"],
        yticklabels=["Normal", "Fraud"],
        ax=ax
    )

    st.pyplot(fig)

    st.divider()

    st.subheader("ROC Curve")

    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()

    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()

    st.pyplot(fig)
    st.divider()
st.subheader("Feature Importance")

try:
    importance = model.coef_[0]

    feature_names = X.columns

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        key=abs,
        ascending=False
    ).head(10)

    fig, ax = plt.subplots()

    sns.barplot(
        x="Importance",
        y="Feature",
        data=importance_df,
        ax=ax
    )

    st.pyplot(fig)

except:
    st.info("Feature importance not available for this model.")
