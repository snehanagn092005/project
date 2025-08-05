import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Set page configuration
st.set_page_config(page_title="Car Mileage Predictor", page_icon="🚗", layout="wide")

# Title and description
st.title("🚗 Car Mileage Predictor")
st.write("Exploratory Data Analysis and MPG Prediction using Linear Regression")

# Load data function
@st.cache_data
def load_data():
    data = pd.read_csv("mtcars.csv")
    return data

df = load_data()

# Sidebar for user inputs
st.sidebar.header("User Input Features")

# Show raw data checkbox
if st.sidebar.checkbox("Show Raw Data"):
    st.subheader("Raw Data")
    st.write(df)

# EDA Section
st.header("Exploratory Data Analysis")

# Data overview
st.subheader("Data Overview")
st.write(f"Dataset contains {df.shape[0]} cars and {df.shape[1]} features")
st.write("First 5 rows:")
st.write(df.head())

# Summary statistics
if st.checkbox("Show Summary Statistics"):
    st.subheader("Summary Statistics")
    st.write(df.describe())

# Visualizations
st.subheader("Data Visualizations")

# Histograms
col1, col2 = st.columns(2)
with col1:
    selected_col_hist = st.selectbox("Select feature for histogram", df.columns[1:])
    fig, ax = plt.subplots()
    sns.histplot(df[selected_col_hist], kde=True, ax=ax)
    ax.set_title(f"Distribution of {selected_col_hist}")
    st.pyplot(fig)

# Scatter plots
with col2:
    selected_x = st.selectbox("X-axis feature", df.columns[1:], index=3)
    selected_y = st.selectbox("Y-axis feature", df.columns[1:], index=0)
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=selected_x, y=selected_y, hue='cyl', ax=ax)
    ax.set_title(f"{selected_y} vs {selected_x}")
    st.pyplot(fig)

# Correlation heatmap
if st.checkbox("Show Correlation Heatmap"):
    st.subheader("Feature Correlation Matrix")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)

# Modeling Section
st.header("MPG Prediction Model")

# Feature selection
st.subheader("Feature Selection")
features = st.multiselect(
    "Select predictors", 
    ['cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear', 'carb'],
    default=['wt', 'disp', 'cyl']
)
target = 'mpg'

# Train-test split
test_size = st.slider("Test Size Proportion", 0.1, 0.5, 0.2)
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

# Model training and evaluation
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Model metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

st.subheader("Model Performance")
st.write(f"R-squared: {r2:.3f}")
st.write(f"Root Mean Squared Error: {rmse:.3f}")

# Show coefficients
st.subheader("Model Coefficients")
coef_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_
})
st.write(coef_df)

# Prediction Section
st.header("Predict MPG for New Car")

st.subheader("Enter Car Specifications")
user_inputs = {}
col1, col2, col3 = st.columns(3)

for i, feature in enumerate(features):
    col = col1 if i % 3 == 0 else col2 if i % 3 == 1 else col3
    min_val = float(df[feature].min())
    max_val = float(df[feature].max())
    default_val = float(df[feature].mean())
    user_inputs[feature] = col.slider(
        f"{feature}", 
        min_value=min_val,
        max_value=max_val,
        value=default_val,
        step=(max_val-min_val)/100
    )

# Create prediction dataframe
input_df = pd.DataFrame([user_inputs])

# Make prediction
if st.button("Predict MPG"):
    prediction = model.predict(input_df)
    st.success(f"Predicted MPG: **{prediction[0]:.1f}**")
    
    # Show actual vs predicted for test set
    st.subheader("Model Visualization")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.7)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel("Actual MPG")
    ax.set_ylabel("Predicted MPG")
    ax.set_title("Actual vs Predicted MPG")
    st.pyplot(fig)

# How to run instructions in sidebar
st.sidebar.header("How to Use")
st.sidebar.info("""
1. Explore raw data and statistics
2. Visualize feature distributions and relationships
3. Select features for prediction model
4. Adjust test size for model validation
5. Enter car specifications and predict MPG
""")

st.sidebar.header("About")
st.sidebar.text("Using mtcars dataset\nLinear Regression Model\nStreamlit EDA App")