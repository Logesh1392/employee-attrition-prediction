import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# App config
st.set_page_config(page_title="Employee Attrition App", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏙️ Home", "📊 Exploratory Data Analysis", "🔮 Predict Attrition"])

# Upload dataset
st.sidebar.subheader("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

# If file uploaded, read it
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

# -------------- HOME ----------------
if page == "🏙️ Home":
    st.title("Employee Attrition Prediction App")
    st.markdown("""
    Welcome to the Employee Attrition Prediction Dashboard!

    ### 🎯 Project Goal:
    Analyze employee-related data and predict whether an employee is likely to leave the organization using machine learning.

    ### 🔍 Features:
    - Upload your dataset
    - Perform Exploratory Data Analysis (EDA)
    - Predict attrition using Random Forest Classifier

    👉 Use the sidebar to navigate between sections.
    """)

# -------------- EDA ----------------
elif page == "📊 Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis (EDA)")

    if uploaded_file is None:
        st.warning("Please upload a dataset to explore.")
    else:
        st.subheader("🔎 Dataset Preview")
        st.dataframe(df.head())

        st.subheader("📏 Dataset Shape")
        st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

        st.subheader("🧠 Data Types")
        st.write(df.dtypes)

        st.subheader("📉 Missing Values")
        st.write(df.isnull().sum())

        st.subheader("📊 Attrition Count")
        if "Attrition" in df.columns:
            fig = px.histogram(df, x="Attrition", color="Attrition", title="Attrition Distribution")
            st.plotly_chart(fig)

        st.subheader("💼 Job Role vs Attrition")
        if "JobRole" in df.columns and "Attrition" in df.columns:
            fig2 = px.histogram(df, x="JobRole", color="Attrition", barmode="group",
                                title="Job Role vs Attrition")
            st.plotly_chart(fig2)

        st.subheader("📈 Age Distribution")
        if "Age" in df.columns:
            fig3 = px.histogram(df, x="Age", nbins=20, title="Age Distribution of Employees")
            st.plotly_chart(fig3)

# -------------- PREDICT ----------------
elif page == "🔮 Predict Attrition":
    st.title("🔮 Predict Employee Attrition")

    if uploaded_file is None:
        st.warning("Please upload a dataset to make predictions.")
    else:
        st.markdown("### ✍️ Select values to predict if the employee will leave")

        # Prepare dataset
        df_model = df.copy()

        # Drop rows where target is missing
        df_model = df_model[df_model["Attrition"].notnull()]

        # Fill missing values for numeric columns with median
        num_cols = df_model.select_dtypes(include=["number"]).columns
        df_model[num_cols] = df_model[num_cols].fillna(df_model[num_cols].median())

        # Fill missing values for categorical columns with mode
        cat_cols = df_model.select_dtypes(include=["object"]).columns
        df_model[cat_cols] = df_model[cat_cols].fillna(df_model[cat_cols].mode().iloc[0])

        # Encode categorical columns
        label_encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            df_model[col] = le.fit_transform(df_model[col])
            label_encoders[col] = le

        # Features and target
        X = df_model.drop("Attrition", axis=1)
        y = df_model["Attrition"]

        # Check for empty data
        if len(X) == 0:
            st.error("No data available after cleaning. Please check the dataset.")
        else:
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train model
            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            # Accuracy
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.success(f"Model trained with accuracy: {accuracy:.2f}")

            # --- Prediction Form ---
            st.markdown("#### 👤 Enter Employee Details:")
            input_data = {}

            for col in X.columns:
                if col in cat_cols:
                    options = df[col].dropna().unique().tolist()
                    input_data[col] = st.selectbox(f"{col}", options)
                else:
                    input_data[col] = st.number_input(
                        f"{col}",
                        float(df[col].min()),
                        float(df[col].max()),
                        float(df[col].mean())
                    )

            # Create input_df
            input_df = pd.DataFrame([input_data])

            # Encode categorical inputs
            for col in input_df.columns:
                if col in label_encoders:
                    input_df[col] = label_encoders[col].transform(input_df[col])

            # Predict
            if st.button("🔍 Predict"):
                prediction = model.predict(input_df)[0]
                pred_label = "Yes" if prediction == 1 else "No"
                st.markdown(f"### 🧾 Predicted Attrition: **{pred_label}**")
