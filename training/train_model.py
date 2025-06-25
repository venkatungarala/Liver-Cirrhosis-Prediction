import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 1. Load data (ensure this CSV is in training/)
df = pd.read_csv('training/liver_data.csv')
df.dropna(inplace=True)

# 2. Split X/y
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']

# 3. Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# 5. Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluate
acc = accuracy_score(y_test, model.predict(X_test))
print(f"✅ Accuracy: {acc*100:.2f}%")

# 7. Save new artifacts under your venv environment
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/rf_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')
print("✅ Saved fresh model + scaler in model/")
