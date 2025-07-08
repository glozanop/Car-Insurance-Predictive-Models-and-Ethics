import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
import json
import os # Import the os library for path manipulation

print("Starting script to retrain baseline model and save components...")

# --- Define Paths ---
data_file_path = "/Users/goyolozano/Desktop/Mini 4/Ethics/Update 4/car_insurance_claim.csv"
output_directory = "/Users/goyolozano/Desktop/Mini 4/Ethics/Update 4/Deliverables"

# --- 1. Load Data ---
try:
    # Use the full path to load the CSV
    df = pd.read_csv(data_file_path)
    print(f"Loaded data from {data_file_path} successfully.")
except FileNotFoundError:
    print(f"Error: Data file not found at {data_file_path}")
    exit() # Exit if the data file isn't found
except Exception as e:
    print(f"An error occurred loading the CSV: {e}")
    exit()

# --- 2. Preprocessing (replicating Update 1 logic) ---

# Exclude specified features
features_to_exclude = ["ID", "GENDER", "MSTATUS", "PARENT1", "EDUCATION"] # Added ID as it's usually excluded
cols_to_drop = [col for col in features_to_exclude if col in df.columns]
df.drop(columns=cols_to_drop, inplace=True)
print(f"Dropped columns: {cols_to_drop}")

# Fill Missing values
print("Filling missing values...")
for col in df.columns:
    if df[col].isnull().any():
      if df[col].dtype in ['float64', 'int64']:
          fill_value = df[col].median()
          df[col].fillna(fill_value, inplace=True)
      else:
          df[col] = df[col].astype('object')
          fill_value = df[col].mode()[0]
          df[col].fillna(fill_value, inplace=True)
print("Missing value filling complete.")


# Separate target variable and features
if 'CLAIM_FLAG' not in df.columns:
     print("Error: Target column 'CLAIM_FLAG' not found.")
     exit()

X = df.drop('CLAIM_FLAG', axis=1)
y = df['CLAIM_FLAG']
print("Separated features (X) and target (y).")

# One-Hot Encode categorical features
print("Applying one-hot encoding...")
X = pd.get_dummies(X, drop_first=True)
feature_names = X.columns.tolist()
print(f"Features after one-hot encoding ({len(feature_names)}): {feature_names[:5]}...")

# Split data into training and testing sets
print("Splitting data into train/test sets (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale numerical features
print("Fitting StandardScaler on training data...")
scaler = StandardScaler()
scaler.fit(X_train)
print("Transforming training data...")
X_train_scaled = scaler.transform(X_train)
print("Scaling complete.")

# --- 3. Train Baseline Logistic Regression Model ---
print("Training baseline Logistic Regression model...")
log_reg_model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
log_reg_model.fit(X_train_scaled, y_train)
print("Model training complete.")

# --- 4. Save Model, Scaler, and Feature Names to Specified Directory ---

# Check if output directory exists, create it if not
if not os.path.exists(output_directory):
    print(f"Output directory '{output_directory}' not found. Creating it...")
    try:
        os.makedirs(output_directory, exist_ok=True)
        print(f"Created output directory: {output_directory}")
    except Exception as e:
        print(f"Error creating output directory: {e}")
        exit()

# Define full paths for output files
model_filename = os.path.join(output_directory, 'lr_model.pkl')
scaler_filename = os.path.join(output_directory, 'scaler.pkl')
features_filename = os.path.join(output_directory, 'feature_names.json')

print(f"Saving trained Logistic Regression model to {model_filename}...")
joblib.dump(log_reg_model, model_filename)

print(f"Saving fitted StandardScaler object to {scaler_filename}...")
joblib.dump(scaler, scaler_filename)

print(f"Saving feature names list to {features_filename}...")
with open(features_filename, 'w') as f:
    json.dump(feature_names, f)

print("\nScript finished successfully!")
print(f"Created files in directory: {output_directory}")
print(f"Files: {os.path.basename(model_filename)}, {os.path.basename(scaler_filename)}, {os.path.basename(features_filename)}")
print("You can now proceed with Step 4 using these files.")