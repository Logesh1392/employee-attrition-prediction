# 🧑‍💼 Employee Attrition Prediction Project

Welcome to the **Employee Attrition Prediction Project**!  
This project offers a complete machine learning pipeline for analyzing employee data, engineering features, training models, and deploying a Streamlit dashboard to predict employee attrition.

---

## 📚 Table of Contents

- [🔍 Project Overview](#-project-overview)
- [✨ Features](#-features)
- [📁 Project Structure](#-project-structure)
- [⚙️ Setup and Installation](#-setup-and-installation)
- [🚀 Usage](#-usage)
  - [1️⃣ Load Raw Data to PostgreSQL](#1️⃣-load-raw-data-to-postgresql)
  - [2️⃣ Clean Data and Push Back](#2️⃣-clean-data-and-push-back)
  - [3️⃣ Feature Engineering](#3️⃣-feature-engineering)
  - [4️⃣ Model Training](#4️⃣-model-training)
  - [5️⃣ Streamlit Dashboard](#5️⃣-streamlit-dashboard)
- [🛠️ Technologies Used](#-technologies-used)
- [📊 Data Description](#-data-description)
- [🤝 Contributing](#-contributing)
- [📝 License](#-license)
- [📬 Contact](#-contact)

---

## 🔍 Project Overview

Employee attrition is a major concern in modern HR management.  
This project helps predict whether an employee is likely to leave an organization using machine learning techniques and an interactive Streamlit dashboard.

It covers everything from raw data loading, cleaning, feature engineering, model training, to deployment of a live app for predictions.

---

## ✨ Features

✅ Load employee data from Excel and store it in a PostgreSQL database  
✅ Clean the data and encode categorical variables  
✅ Perform feature engineering to create meaningful new columns  
✅ Train models such as Random Forest, Logistic Regression, XGBoost, and LightGBM  
✅ Launch an interactive Streamlit dashboard for:
- Exploratory Data Analysis (EDA)
- Real-time Attrition Prediction

✅ Export intermediate datasets as CSV for external use

---

## 📁 Project Structure

```
E:\EmployeeProject
├── 1_push_to_postgres.py          # Load Excel data and push to PostgreSQL
├── 2_clean_and_push_back.py      # Clean and save data back to DB and CSV
├── 4_feature_engineering.py      # Generate new features and save them
├── 5_model_training.ipynb        # Model training and evaluation (Jupyter)
├── 6_streamlit_main.py           # Streamlit dashboard for EDA & Prediction
├── employee_with_features.csv    # Final cleaned and feature-enhanced dataset
├── Employee-Attrition.xlsx       # Original Excel dataset (not included)
```

---

## ⚙️ Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/employee-attrition-prediction.git
cd employee-attrition-prediction
```

### 2. Install required packages

```bash
pip install -r requirements.txt
```

### 3. Setup PostgreSQL

- Install and run PostgreSQL on your machine.
- Create a database named `test`.
- Update credentials in Python files if needed.

### 4. Prepare your data

- Place the original dataset `Employee-Attrition.xlsx` in the project root folder or update the script paths accordingly.

---

## 🚀 Usage

### 1️⃣ Load Raw Data to PostgreSQL

```bash
python 1_push_to_postgres.py
```
- Reads the Excel file and inserts the raw data into the `raw_employee` table.

---

### 2️⃣ Clean Data and Push Back

```bash
python 2_clean_and_push_back.py
```
- Cleans unnecessary columns  
- Encodes categorical variables  
- Saves cleaned data to PostgreSQL and CSV  

---

### 3️⃣ Feature Engineering

```bash
python 4_feature_engineering.py
```
- Adds new columns such as:
  - `EngagementScore`
  - `TenureLevel`
  - `OverTime_Flag`

---

### 4️⃣ Model Training

```bash
jupyter notebook 5_model_training.ipynb
```
- Train models like:
  - RandomForestClassifier
  - Logistic Regression
  - XGBoost
  - LightGBM  
- Evaluate performance using metrics like accuracy, precision, recall.

---

### 5️⃣ Streamlit Dashboard

```bash
streamlit run 6_streamlit_main.py
```
- Use the sidebar to:
  - Upload a new CSV  
  - Explore the dataset visually (EDA)  
  - Predict employee attrition using the trained model  

---

## 🛠️ Technologies Used

- Python 3.x  
- Pandas, NumPy – Data manipulation  
- Scikit-learn – ML model building  
- XGBoost, LightGBM – Advanced classifiers  
- SQLAlchemy, psycopg2 – PostgreSQL integration  
- Streamlit – Interactive dashboard  
- Plotly – Beautiful visualizations  

---

## 📊 Data Description

The dataset contains employee-level information such as:

| Feature           | Description                             |
|-------------------|-----------------------------------------|
| Age               | Employee age                            |
| Gender            | Male / Female                           |
| Department        | HR / Sales / R&D                        |
| JobRole           | Employee’s designation                  |
| BusinessTravel    | Travel frequency                        |
| OverTime          | Whether employee works overtime         |
| JobSatisfaction   | Satisfaction rating (1–4)               |
| PerformanceRating | Performance score (1–4)                 |
| YearsAtCompany    | Years spent at the company              |
| TotalWorkingYears | Total industry experience               |
| Attrition         | 1 = Yes (left), 0 = No (stayed)         |

---


---

## 📬 Contact

For any questions or feedback, feel free to reach out:

- [lokesh.waran1392@gmail.com](mailto:lokesh.waran1392@gmail.com)  
– [Logesh1392](https://github.com/Logesh1392)

---

Thank you for exploring the Employee Attrition Prediction Project! 🚀
