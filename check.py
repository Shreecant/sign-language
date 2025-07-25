import pandas as pd

# Load the dataset
df = pd.read_csv("my_asl_data.csv")

# Show top few rows
print("🔍 First 5 Rows:")
print(df.head())

# Check how many samples per label
print("\n📊 Label counts:")
print(df['label'].value_counts())
