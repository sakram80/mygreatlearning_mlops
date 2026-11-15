import os
import pandas as pd

# Path to the dataset inside the repo
data_path = "tourism_project/data/tourism.csv"

# 1. Verify the file exists
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found at '{data_path}'. Please ensure tourism.csv is placed in tourism_project/data/.")

# 2. Read the CSV
df = pd.read_csv(data_path)

# 3. Check for expected columns
expected_columns = [
    'CustomerID', 'ProdTaken', 'Age', 'TypeofContact', 'CityTier',
    'DurationOfPitch', 'Occupation', 'Gender', 'NumberOfPersonVisiting',
    'NumberOfFollowups', 'ProductPitched', 'PreferredPropertyStar',
    'MaritalStatus', 'NumberOfTrips', 'Passport', 'PitchSatisfactionScore',
    'OwnCar', 'NumberOfChildrenVisiting', 'Designation', 'MonthlyIncome'
]

missing_cols = [col for col in expected_columns if col not in df.columns]

if missing_cols:
    raise ValueError(f"Missing expected columns: {missing_cols}")
else:
    print("Column validation passed: All expected columns are present.")

# 4. Print a short summary
print("\n--- Dataset Summary ---")
print(f"Total Rows: {df.shape[0]}")
print(f"Total Columns: {df.shape[1]}")
print(f"Target Distribution (ProdTaken):\n{df['ProdTaken'].value_counts(dropna=False).to_dict()}")
print(f"Total Missing Values: {df.isnull().sum().sum()}")
