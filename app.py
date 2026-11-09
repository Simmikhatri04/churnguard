import os, json
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(page_title="ChurnGuard", page_icon="📉", layout="wide")

@st.cache_resource
def load_assets():
    model = joblib.load("churn_model.pkl")
    cols = joblib.load("columns.pkl")
    threshold = json.load(open("config.json"))["threshold"]
    return model, cols, threshold

@st.cache_data
def load_data():
    return pd.read_csv("clean_churn.csv")

model, cols, THRESHOLD = load_assets()
df = load_data()

def to_features(data):
    return pd.get_dummies(data).reindex(columns=cols, fill_value=0)

def risk_label(p):
    if p >= THRESHOLD: return "High"
    if p >= 0.30: return "Medium"
    return "Low"

def suggest(row):
    tips = []
    if row["Contract"] == "Month-to-month":
        tips.append("Offer a discount for a 1-year contract")
    if row["InternetService"] == "Fiber optic":
        tips.append("Run a service-quality check-in")
    if row["tenure"] < 12:
        tips.append("Add onboarding support and a welcome offer")
    return "; ".join(tips) or "Keep engaged with loyalty rewards"

st.title("📉 ChurnGuard: Telecom Churn Prediction")
st.caption("Find customers likely to leave, understand why, and decide how to keep them.")

t1, t2, t3, t4, t5 = st.tabs(
    ["Overview", "Predict", "Batch Scoring", "Model Performance", "Key Drivers"])

# ---------- Overview ----------
with t1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Customers", f"{len(df):,}")
    c2.metric("Churn rate", f"{df['Churn'].mean()*100:.1f}%")
    c3.metric("Avg monthly charge", f"${df['MonthlyCharges'].mean():.2f}")
    l, r = st.columns(2)
    by_c = df.groupby("Contract")["Churn"].mean().reset_index()
    l.plotly_chart(px.bar(by_c, x="Contract", y="Churn",
                          title="Churn rate by contract"), use_container_width=True)
    r.plotly_chart(px.histogram(df, x="tenure", color="Churn", barmode="overlay",
                                title="Tenure vs churn"), use_container_width=True)

# ---------- Predict one customer ----------
with t2:
    a, b = st.columns(2)
    tenure = a.slider("Tenure (months)", 0, 72, 12)
    monthly = a.slider("Monthly charges ($)", 18, 120, 70)
    contract = a.selectbox("Contract", df["Contract"].unique())
    internet = b.selectbox("Internet service", df["InternetService"].unique())
    payment = b.selectbox("Payment method", df["PaymentMethod"].unique())
    senior = b.selectbox("Senior citizen", [0, 1])

    if st.button("Predict churn risk"):
        row = df.drop(columns="Churn").iloc[[0]].copy()
        for c in row.columns:
            row[c] = df[c].median() if pd.api.types.is_numeric_dtype(df[c]) else df[c].mode()[0]
        row["tenure"], row["MonthlyCharges"] = tenure, monthly
        row["TotalCharges"] = tenure * monthly
        row["Contract"], row["InternetService"] = contract, internet
        row["PaymentMethod"], row["SeniorCitizen"] = payment, senior
        p = model.predict_proba(to_features(row))[0][1]
        st.metric("Churn probability", f"{p*100:.1f}%")
        lvl = risk_label(p)
        (st.error if lvl == "High" else st.warning if lvl == "Medium" else st.success)(
            f"{lvl} risk")
        st.info("Suggested action: " + suggest(row.iloc[0]))

# ---------- Batch scoring ----------
with t3:
    st.write("Upload a CSV in the Telco format to score many customers at once.")
    up = st.file_uploader("CSV file", type="csv")
    if up:
        raw = pd.read_csv(up)
        d = raw.copy()
        d["TotalCharges"] = pd.to_numeric(d["TotalCharges"], errors="coerce").fillna(0)
        d = d.drop(columns=[c for c in ["customerID", "Churn"] if c in d.columns])
        p = model.predict_proba(to_features(d))[:, 1]
        out = raw.copy()
        out["ChurnProbability"] = p.round(3)
        out["Risk"] = [risk_label(x) for x in p]
        out["SuggestedAction"] = d.apply(suggest, axis=1)
        st.write(out["Risk"].value_counts())
        top = out.sort_values("ChurnProbability", ascending=False)
        st.dataframe(top.head(50), use_container_width=True)
        st.download_button("Download scored customers",
                           top.to_csv(index=False), "scored_customers.csv")

# ---------- Model performance ----------
with t4:
    st.subheader("Model comparison")
    st.dataframe(pd.read_csv("model_comparison.csv"), use_container_width=True)
    st.write(f"Final model: tuned Random Forest, decision threshold **{THRESHOLD}** "
             "chosen to favor recall, since missing a churner costs more than a "
             "wasted retention offer.")

# ---------- Key drivers ----------
with t5:
    imp = pd.Series(model.feature_importances_, index=cols).nlargest(10)
    st.subheader("Top 10 features (Random Forest importance)")
    st.bar_chart(imp)
    if os.path.exists("shap_summary.png"):
        st.subheader("SHAP summary")
        st.image("shap_summary.png")