import pandas as pd
from sqlalchemy import create_engine

# Load the cleaned CSV
db_user = "postgres"
db_password = "admin123"
db_host = "localhost"
db_port = "5432"
db_name = "test"

engine = create_engine(f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")
df = pd.read_sql('SELECT * FROM employee_cleaned', engine)

# 1. TenureCategory
def tenure_category(years):
    if years < 3:
        return 'Short-term'
    elif years <= 7:
        return 'Mid-term'
    else:
        return 'Long-term'

df['TenureCategory'] = df['YearsAtCompany'].apply(tenure_category)

# 2. OverTimeBinary: Yes -> 1, No -> 0
df['OverTimeBinary'] = df['OverTime'].map({'Yes': 1, 'No': 0})

# 3. EngagementScore: Combine JobSatisfaction + JobInvolvement (scale: 1-4)
df['EngagementScore'] = df['JobSatisfaction'] + df['JobInvolvement']

# Preview
print(df[['YearsAtCompany', 'TenureCategory', 'OverTime', 'OverTimeBinary', 
          'JobSatisfaction', 'JobInvolvement', 'EngagementScore']].head())

# Optional: Save to new CSV
df.to_csv("employee_with_features.csv", index=False)

# Push to PostgreSQL (update credentials accordingly)
df.to_sql('employee_features', engine, if_exists='replace', index=False)

print("\n✅ Feature Engineering complete. CSV saved as 'employee_with_features.csv'.")
