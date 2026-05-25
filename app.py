import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Page Config

st.set_page_config(page_title="Salary Prediction", layout="centered")

# Load CSS

def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# Title

st.markdown("""
    <div class="card">
    <h1>Linear Regression</h1>
    <p>Predict <b>Salary</b> from <b>Years of Experience</b> using Linear Regression...</p>
    </div>
""", unsafe_allow_html=True)

# Load Data

@st.cache_data
def load_data():
    df = pd.read_csv("Salary_Data.csv")
    return df

df = load_data()

# Dataset Preview

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df.head())
st.markdown('</div>', unsafe_allow_html=True)

# Prepare Data

x = df[["YearsExperience"]]
y = df["Salary"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Train Model

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

# Metrics

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

adjusted_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - 2)

# Visualization

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Years of Experience vs Salary")

fig, ax = plt.subplots()

ax.scatter(df["YearsExperience"], df["Salary"], alpha=0.6)

ax.plot(
    df["YearsExperience"],
    model.predict(scaler.transform(x)),
    color="red"
)

ax.set_xlabel("Years of Experience")
ax.set_ylabel("Salary")

st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)

# Performance

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Model Performance")

c1, c2 = st.columns(2)
c1.metric("MAE", f"{mae:.2f}")
c2.metric("RMSE", f"{rmse:.2f}")

c3, c4 = st.columns(2)
c3.metric("R²", f"{r2:.3f}")
c4.metric("Adjusted R²", f"{adjusted_r2:.3f}")

st.markdown('</div>', unsafe_allow_html=True)

# Coefficient and Intercept

st.markdown(f"""
    <div class="card">
    <h3>Model Intercept and Coefficient</h3>
    <p>
    <b>Coefficient:</b> {model.coef_[0]:.3f}<br>
    <b>Intercept:</b> {model.intercept_:.3f}
    </p>
    </div>
""", unsafe_allow_html=True)

# Prediction

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Predict Salary")

experience = st.slider(
    "Years of Experience",
    float(df.YearsExperience.min()),
    float(df.YearsExperience.max()),
    5.0
)

input_df = pd.DataFrame(
    [[experience]],
    columns=["YearsExperience"]
)

salary = model.predict(
    scaler.transform(input_df)
)[0]

st.markdown(
    f'<div class="prediction-box">Predicted Salary: ₹ {salary:,.2f}</div>',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)