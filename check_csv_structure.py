import pandas as pd

try:
    # Try reading only the first few lines
    df = pd.read_csv("my_asl_data.csv", nrows=5)
    print("✅ File loaded successfully. Columns:")
    print(df.columns.tolist())
    print(df.head())
except Exception as e:
    print("❌ Error reading file:", e)
