import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

# Set page config with modern design theme
st.set_page_config(page_title="Data Analysis & Prediction", layout="wide")

# Initialize session state for navigation
if "page" not in st.session_state:
    st.session_state.page = "home"

# Custom CSS for Magma Theme
st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #1e1e2e, #2b2b3c);
            color: white;
            font-family: Arial, sans-serif;
        }
        .stButton>button {
            background: linear-gradient(90deg, #ff4500, #ff1493);
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 10px;
            transition: 0.3s;
        }
        .stButton>button:hover {
            transform: scale(1.1);
        }
        .stDataFrame {
            border-radius: 10px;
            background-color: #2e2e3e;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Navigation Function
def navigate_to(page):
    st.session_state.page = page

# Sidebar Navigation
st.sidebar.title("🔗 Navigation")
if st.sidebar.button("🏠 Home"):
    navigate_to("home")
if st.sidebar.button("📊 Data Overview"):
    navigate_to("data_overview")
if st.sidebar.button("🛠 Data Cleaning"):
    navigate_to("data_cleaning")
if st.sidebar.button("🤖 Predictive Modeling"):
    navigate_to("modeling")

# Home Page
if st.session_state.page == "home":
    st.title("🔥 Data Analysis & Predictive Modeling App")
    st.write("🚀 Upload a dataset to get started!")

    # Upload dataset
    uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])
    if uploaded_file:
        file_extension = uploaded_file.name.split(".")[-1]
        df = pd.read_csv(uploaded_file) if file_extension == "csv" else pd.read_excel(uploaded_file)
        st.session_state.df = df  # Save the dataset in session state
        st.success("✅ Dataset Uploaded Successfully!")

        # Navigation button
        if st.button("Go to Data Overview ➡️"):
            navigate_to("data_overview")

# Data Overview Page
elif st.session_state.page == "data_overview":
    st.title("📊 Dataset Overview")
    
    if "df" in st.session_state:
        df = st.session_state.df
        st.subheader("📋 Dataset Preview")
        st.dataframe(df.head())

        # Search for specific column or row information
        st.subheader("🔎 Search Dataset")
        search_col = st.text_input("Enter column name to search")
        if search_col and search_col in df.columns:
            st.write(df[search_col])
        elif search_col:
            st.write("❌ Column not found")

        search_row = st.number_input("Enter row index to search", min_value=0, max_value=len(df)-1, step=1)
        if search_row >= 0 and search_row < len(df):
            st.write(df.iloc[search_row])

        # Navigation button
        if st.button("Next: Data Cleaning ➡️"):
            navigate_to("data_cleaning")
    else:
        st.warning("⚠️ No dataset uploaded. Please go back to Home.")

# Data Cleaning Page
elif st.session_state.page == "data_cleaning":
    st.title("🛠 Data Cleaning & Transformation")

    if "df" in st.session_state:
        df = st.session_state.df.copy()

        # Missing Values Handling
        missing_threshold = st.slider("Select missing value threshold for column removal", 0, 100, 50)
        df_cleaned = df.dropna(thresh=len(df) * (missing_threshold / 100), axis=1)
        st.session_state.df_cleaned = df_cleaned  # Save cleaned dataset
        st.write("✅ Columns after cleaning:", df_cleaned.columns.tolist())

        # Encoding categorical data
        categorical_cols = df_cleaned.select_dtypes(include=["object"]).columns
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df_cleaned[col] = le.fit_transform(df_cleaned[col].astype(str))
            label_encoders[col] = le
        st.write("📝 Categorical Columns Encoded:", categorical_cols.tolist())

        # Feature Scaling
        scaler = StandardScaler()
        numerical_cols = df_cleaned.select_dtypes(include=["number"]).columns
        df_cleaned[numerical_cols] = scaler.fit_transform(df_cleaned[numerical_cols])
        st.write("📊 Numerical Columns Scaled:", numerical_cols.tolist())

        # Save cleaned data
        st.session_state.df_cleaned = df_cleaned

        # Navigation button
        if st.button("Next: Predictive Modeling ➡️"):
            navigate_to("modeling")
    else:
        st.warning("⚠️ No dataset uploaded. Please go back to Home.")

# Predictive Modeling Page
elif st.session_state.page == "modeling":
    st.title("🤖 Predictive Analysis")

    if "df_cleaned" in st.session_state:
        df_cleaned = st.session_state.df_cleaned

        target_col = st.selectbox("🎯 Select Target Column", df_cleaned.columns)
        if target_col:
            X = df_cleaned.drop(columns=[target_col])
            y = df_cleaned[target_col]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if len(y.unique()) > 5:  # Regression
                model = RandomForestRegressor()
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                mse = mean_squared_error(y_test, predictions)
                st.write(f"📉 Mean Squared Error: {mse:.4f}")
            else:  # Classification
                model = RandomForestClassifier()
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)
                st.write(f"✅ Accuracy: {accuracy:.4f}")

        # Restart button
        if st.button("🔄 Start Over"):
            navigate_to("home")
    else:
        st.warning("⚠️ No cleaned dataset found. Please go back to Data Cleaning.")

