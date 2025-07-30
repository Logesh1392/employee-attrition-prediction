import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# ----------------- Helper Function -------------------
def safe_dataframe_display(dataframe, caption=None):
    """
    Safely displays a pandas DataFrame in Streamlit,
    avoiding SettingWithCopyWarning during Arrow conversion.
    """
    try:
        dataframe = dataframe.copy()
        for col in dataframe.columns:
            if pd.api.types.is_integer_dtype(dataframe[col]) and dataframe[col].isnull().any():
                dataframe.loc[:, col] = dataframe[col].astype(float)
            elif dataframe[col].dtype == "object":
                dataframe.loc[:, col] = dataframe[col].astype(str)

        st.dataframe(dataframe, use_container_width=True)
    except Exception as e:
        st.error(f"⚠️ Error displaying dataframe: {e}")

# ----------------- Load Model -------------------
model = joblib.load("best_lgb_model.pkl")

expected_features = [
    'Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome',
    'Education', 'EducationField', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate',
    'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus',
    'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike',
    'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
    'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
    'YearsSinceLastPromotion', 'YearsWithCurrManager', 'Over18', 'StandardHours'
]

# ----------------- Streamlit Config -------------------
st.set_page_config(page_title="Employee Attrition App", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏙️ Home", "📊 Exploratory Data Analysis", "🔮 Predict Attrition", "📈 Model Insights"])

st.sidebar.subheader("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])
# Message before and after upload
if uploaded_file is None:
    st.sidebar.markdown(
        """
        <style>
        .blinking {
            animation: blinker 1s linear infinite;
            color: red;
            font-weight: bold;
        }
        @keyframes blinker {
            50% { opacity: 0; }
        }
        </style>
        <p class="blinking">⬆️ Please upload your dataset to begin analysis.</p>
        """,
        unsafe_allow_html=True
    )
else:
    st.sidebar.success("✅ Thank you for uploading the dataset!")

df = pd.read_csv(uploaded_file) if uploaded_file else pd.DataFrame()

# ----------------- Home Page -------------------
if page == "🏙️ Home":
    st.title("Employee Attrition Prediction App")
    st.markdown("""
    Welcome to the Employee Attrition Prediction Dashboard!

    ### 🎯 Project Goal:
    Analyze employee-related data and predict whether an employee is likely to leave the organization using machine learning.

    ### 🔍 Features:
    - Upload your dataset
    - Perform Exploratory Data Analysis (EDA)
    - Predict attrition using a trained LightGBM model
    - View classification metrics and insights

    👉 Use the sidebar to navigate between sections.
    """)

# ----------------- EDA Page -------------------
elif page == "📊 Exploratory Data Analysis" and not df.empty:
    st.title("📊 Exploratory Data Analysis (EDA)")

    st.subheader("🔎 Dataset Preview")
    safe_dataframe_display(df.head())

    st.subheader("📏 Dataset Shape")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    st.subheader("🧠 Data Types")
    safe_dataframe_display(df.dtypes.reset_index().rename(columns={0: 'DataType', 'index': 'Column'}))

    st.subheader("📉 Missing Values")
    safe_dataframe_display(df.isnull().sum().reset_index().rename(columns={0: 'MissingValues', 'index': 'Column'}))

    if "Attrition" in df.columns:
        st.subheader("📊 Attrition Count")
        st.plotly_chart(px.histogram(df, x="Attrition", color="Attrition", title="Attrition Distribution"))

    if "JobRole" in df.columns and "Attrition" in df.columns:
        st.subheader("💼 Job Role vs Attrition")
        st.plotly_chart(px.histogram(df, x="JobRole", color="Attrition", barmode="group"))

    if "Age" in df.columns:
        st.subheader("📈 Age Distribution")
        st.plotly_chart(px.histogram(df, x="Age", nbins=20))

    if "Gender" in df.columns:
        st.subheader("👫 Gender Distribution")
        st.plotly_chart(px.pie(df, names="Gender"))

    if "Department" in df.columns:
        st.subheader("🏢 Department Distribution")
        st.plotly_chart(px.pie(df, names="Department"))

    if "MonthlyIncome" in df.columns and "JobRole" in df.columns:
        st.subheader("📦 Monthly Income by Job Role")
        st.plotly_chart(px.box(df, x="JobRole", y="MonthlyIncome"))

    st.subheader("🧪 Correlation Heatmap")
    corr = df.select_dtypes(include='number').corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, cmap='coolwarm', annot=False, ax=ax)
    st.pyplot(fig)

# ----------------- Prediction Page -------------------
elif page == "🔮 Predict Attrition":
    st.header("📂 Predict for Multiple Employees (CSV Upload)")
    uploaded_file = st.file_uploader("Upload employee data CSV", type=["csv"])

    def preprocess_input(df):
        for col in expected_features:
            if col not in df.columns:
                df[col] = "Unknown" if col in [
                    'BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole',
                    'MaritalStatus', 'OverTime', 'Over18'] else 0
        df = df[expected_features]
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].astype('category')
        return df

    def predict_attrition(input_df, model):
        processed_df = preprocess_input(input_df.copy())
        preds = model.predict(processed_df)
        return ["Yes" if p == 1 else "No" for p in preds]

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            predictions = predict_attrition(input_df, model)
            input_df["Attrition_Prediction"] = predictions

            st.success("✅ Prediction complete!")
            safe_dataframe_display(input_df, caption="Prediction Results")

            csv = input_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Result", data=csv, file_name="predicted_attrition.csv", mime="text/csv")

            st.header("🧍 Single Employee Prediction (Form with Default Values)")
            default_row = input_df.sample(1).drop(columns=["Attrition_Prediction"]).squeeze()

            with st.form("single_prediction_form"):
                st.write("### Fill or Edit Employee Details:")
                input_data = {}
                for col in expected_features:
                    if col in ['OverTime', 'Gender', 'MaritalStatus', 'Department', 'EducationField', 'JobRole', 'BusinessTravel', 'Over18']:
                        unique_vals = input_df[col].dropna().unique().tolist()
                        default_value = default_row[col] if col in default_row else unique_vals[0]
                        input_data[col] = st.selectbox(col, options=unique_vals, index=unique_vals.index(default_value) if default_value in unique_vals else 0)
                    else:
                        default_value = float(default_row[col]) if col in default_row else 0.0
                        input_data[col] = st.number_input(col, value=default_value, format="%.2f")

                submitted = st.form_submit_button("🔍 Predict Attrition")
                if submitted:
                    form_df = pd.DataFrame([input_data])
                    prediction = predict_attrition(form_df, model)[0]
                    st.success(f"🔮 Predicted Attrition: **{prediction}**")
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")

# ----------------- Insights Page -------------------
elif page == "📈 Model Insights" and not df.empty:
    st.title("📈 Model Insights and Evaluation")
    st.markdown("Evaluation is based on the uploaded data with 'Attrition' column.")

    if "Attrition" in df.columns:
        label_mapping = {v: i for i, v in enumerate(df['Attrition'].dropna().unique())}
        y_true = df['Attrition'].map(label_mapping)
        X = df.drop(columns=["Attrition"])

        for col in expected_features:
            if col not in X.columns:
                X[col] = "Unknown" if col in ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime', 'Over18'] else 0
        X = X[expected_features]

        for col in X.columns:
            if X[col].dtype == 'object':
                mapping = {v: i for i, v in enumerate(X[col].dropna().unique())}
                X[col] = X[col].map(mapping)
            else:
                X[col] = X[col].fillna(X[col].median())

        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]

        st.subheader("📋 Classification Report")
        report = classification_report(y_true, y_pred, output_dict=True)
        safe_dataframe_display(pd.DataFrame(report).transpose())

        st.subheader("🧩 Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual", color="Count"))
        st.plotly_chart(fig_cm)

        st.subheader("📈 ROC Curve")
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        fig_roc = px.area(
            x=fpr, y=tpr,
            title=f'ROC Curve (AUC = {roc_auc:.2f})',
            labels=dict(x='False Positive Rate', y='True Positive Rate'),
        )
        fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
        st.plotly_chart(fig_roc)
    else:
        st.warning("Please upload a dataset containing the 'Attrition' column to view model insights.")
