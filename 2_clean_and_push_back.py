import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder

# 1. Database connection variables
db_user = "postgres"
db_password = "admin123"
db_host = "localhost"
db_port = "5432"
db_name = "test"

# 2. Create PostgreSQL connection engine
engine = create_engine(f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")

# 3. Load raw data from PostgreSQL
df = pd.read_sql("SELECT * FROM employee_raw", engine)

# 4. Drop unnecessary columns if present
drop_cols = ['EmployeeNumber', 'EmployeeCount', 'Over18', 'StandardHours']
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

# 5. Encode categorical columns
label_enc = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = label_enc.fit_transform(df[col])

# 6. Save cleaned data back to PostgreSQL
df.to_sql("employee_cleaned", engine, if_exists="replace", index=False)

print("✅ Cleaned data saved to PostgreSQL.")

# 7. Save cleaned data to project folder as CSV
project_path = "E:\EmployeeProject/employee_cleaned.csv"
df.to_csv(project_path, index=False)

print("✅ Cleaned data saved to PostgreSQL and local project folder.")
