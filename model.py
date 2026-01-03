import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv("heartdisease.csv")

# Display data information
print("Dataset Information:")
print(f"Total samples: {data.shape[0]}")
print(f"Features: {data.shape[1] - 1}")
print("Target distribution:")
print(data["target"].value_counts())
print("\n")

# Define features and target
X = data.drop(columns=["target"])
y = data["target"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create preprocessing pipeline
imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()

# Apply preprocessing to training data
X_train_imputed = imputer.fit_transform(X_train)
X_train_scaled = scaler.fit_transform(X_train_imputed)

# Apply preprocessing to test data
X_test_imputed = imputer.transform(X_test)
X_test_scaled = scaler.transform(X_test_imputed)

# Define models to evaluate
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
}

# Train and evaluate models
best_model = None
best_accuracy = 0
best_model_name = ""

print("Model Evaluation:")
print("-" * 50)

for name, model in models.items():
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.4f}")
    
    # Print classification report
    print(f"\nClassification Report for {name}:")
    print(classification_report(y_test, y_pred))
    
    # Track best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name

print("-" * 50)
print(f"Best Model Selected: {best_model_name} with Accuracy: {best_accuracy:.4f}")

# Create a dictionary containing all preprocessing objects and the model
pipeline = {
    'imputer': imputer,
    'scaler': scaler,
    'model': best_model,
    'feature_names': list(X.columns)
}

# Save the entire pipeline
pickle.dump(pipeline, open("model.pkl", "wb"))
print("Model pipeline saved to model.pkl")

# Feature importance (if Random Forest was selected)
if best_model_name == "Random Forest":
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)