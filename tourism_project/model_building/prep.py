import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load the dataset
df = pd.read_csv("tourism_project/data/tourism.csv")

# 2. Remove unnecessary columns and fix minor typos
cols_to_drop = ['CustomerID', 'Unnamed: 0']
df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].replace({'Fe Male': 'Female'})

# 3. Split into X and y, then into train and test sets
target_col = 'ProdTaken'
X = df.drop(columns=[target_col])
y = df[target_col]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Save the four split files in the root folder for GitHub Actions artifacts
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Saved Xtrain.csv, Xtest.csv, ytrain.csv, and ytest.csv successfully.")
