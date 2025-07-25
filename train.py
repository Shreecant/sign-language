import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Step 1: Load your dataset
df = pd.read_csv("my_asl_data.csv")
print(f"✅ Loaded dataset with {len(df)} samples and {df.shape[1]} columns")

# Step 2: Split features and labels
X = df.drop("label", axis=1)
y = df["label"]

# Step 3: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Step 5: Evaluate model
y_pred = clf.predict(X_test)
print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n🧠 Classification Report:\n", classification_report(y_test, y_pred))

# Step 6: Save model
joblib.dump(clf, "asl_model.pkl")
print("\n💾 Model saved as 'asl_model.pkl'")
