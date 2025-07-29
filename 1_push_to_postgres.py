import pandas as pd
from sqlalchemy import create_engine

# 1. Load Excel file
df = pd.read_excel("E:\EmployeeProject\Employee-Attrition.xlsx")
df.columns = df.columns.str.strip()

# 2. PostgreSQL credentials
db_user = "postgres"
db_password = "admin123"
db_host = "localhost"
db_port = "5432"
db_name = "test"

# 3. Build connection engine
engine = create_engine(f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")

# 4. Push to PostgreSQL
df.to_sql("employee_raw", engine, if_exists="replace", index=False)

print("✅ Raw data pushed to PostgreSQL.")
