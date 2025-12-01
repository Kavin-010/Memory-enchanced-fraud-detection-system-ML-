import pandas as pd
from sklearn.model_selection import train_test_split

# Load your full dataset
df = pd.read_csv("creditcard.csv")   # or whatever your dataset name is
print(f"Original dataset: {df.shape}")

# Split the dataset (80% train, 20% test)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Class'])

# Save to new CSV files
train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)

print(f"Train set: {train_df.shape}")
print(f"Test set: {test_df.shape}")
print("[INFO] train.csv and test.csv have been created successfully!")
