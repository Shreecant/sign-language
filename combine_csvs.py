import pandas as pd
import os

data_dir = "data"
combined_df = pd.DataFrame()

for file in os.listdir(data_dir):
    if file.endswith(".csv"):
        file_path = os.path.join(data_dir, file)
        try:
            df = pd.read_csv(file_path)
            if not df.empty:
                combined_df = pd.concat([combined_df, df], ignore_index=True)
                print(f"✔ Loaded {file} with {len(df)} samples")
            else:
                print(f"⚠ Skipped empty file: {file}")
        except Exception as e:
            print(f"❌ Error reading {file}: {e}")

if not combined_df.empty:
    combined_df.to_csv("my_asl_data.csv", index=False)
    print(f"✅ Combined all valid files into my_asl_data.csv with {len(combined_df)} total samples")
else:
    print("❗ No valid data found. Please record some samples first.")
