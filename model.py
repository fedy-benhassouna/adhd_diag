import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np


# Load dataset
df = pd.read_csv("adhd_dataset_with_noise_and_complexity.csv")

# Add random noise to features
df["Not Looking (times)"] += np.random.normal(0, 2, df.shape[0])
df["Useless Touches"] += np.random.normal(0, 2, df.shape[0])

# Ensure values are non-negative
df["Not Looking (times)"] = df["Not Looking (times)"].clip(lower=0)
df["Useless Touches"] = df["Useless Touches"].clip(lower=0)

# Features (X) and target (y)
X = df[["Not Looking (times)", "Useless Touches","Avg Time Between Touches"]]
y = df["ADHD/NOT ADHD"]

# Split into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))