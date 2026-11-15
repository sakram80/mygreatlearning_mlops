import os
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load the dataset from the repository data folder
data_path = "tourism_project/data/tourism.csv"
df = pd.read_csv(data_path)

# 2. Remove unnecessary columns
# Drop identifier or auto-generated index columns that do not help prediction
cols_to_drop = ['CustomerID', 'Unnamed: 0']
df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

# Clean minor formatting inconsistencies if present
if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].replace({'Fe Male': 'Female'})

# 3. Split the data into training and testing sets (80% train, 20% test)
target_col = 'ProdTaken'

if target_col in df.columns:
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df[target_col]
    )
else:
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 4. Save the split files locally as CSV files
output_dir = "tourism_project/data"
os.makedirs(output_dir, exist_ok=True)

train_path = os.path.join(output_dir, "train.csv")
test_path = os.path.join(output_dir, "test.csv")

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print("Data preparation completed successfully.")
print(f"Training set saved to {train_path} with shape {train_df.shape}")
print(f"Testing set saved to {test_path} with shape {test_df.shape}")
