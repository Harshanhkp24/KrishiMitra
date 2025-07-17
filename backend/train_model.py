import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Sample dummy dataset
data = {
    "soil_type": ["Loamy", "Sandy", "Clay", "Loamy", "Sandy"],
    "rainfall": [300, 200, 100, 350, 250],
    "temperature": [25, 30, 20, 24, 29],
    "crop": ["Wheat", "Rice", "Millet", "Wheat", "Rice"]
}

df = pd.DataFrame(data)

# Convert categorical to numeric
df["soil_type"] = df["soil_type"].astype("category").cat.codes
df["crop"] = df["crop"].astype("category")
crop_labels = dict(enumerate(df["crop"].cat.categories))
df["crop"] = df["crop"].cat.codes

# Split features and target
X = df[["soil_type", "rainfall", "temperature"]]
y = df["crop"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model and label mapping
joblib.dump(model, "crop_model.pkl")
joblib.dump(crop_labels, "label_map.pkl")

print("✅ Model trained and saved as 'crop_model.pkl'")
