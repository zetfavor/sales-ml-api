import joblib
from pathlib import Path
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Import our custom source-code modules
from src.data_loader import load_config, load_raw

print("Starting final model training...")

# 1. Define Project Root
# We use this to build absolute paths
ROOT = Path(__file__).resolve().parent

# 2. Load the "golden" config
config = load_config()

# 3. Load the *entire* raw dataset
df = load_raw("sample_sales.csv")
X = df.drop("target", axis=1)
y = df["target"]

print(f"Loaded {len(df)} total samples.")

# 4. Apply SMOTE to 100% of the data
# (We only do this if the config says so)
if config['training']['smote']:
    print("Applying SMOTE to all data...")
    smote = SMOTE(random_state=config['training']['random_seed'])
    X_final, y_final = smote.fit_resample(X, y)
    print(f"Data resampled. New shape: {X_final.shape}")
else:
    X_final, y_final = X, y

# 5. Create the model with our "golden" parameters
model = XGBClassifier(
    random_state=config['training']['random_seed'],
    **config['training']['model']['params']  # This loads all the params from the config
)

# 6. Train the model on ALL available data
print("Training final model on all data...")
model.fit(X_final, y_final)

# 7. Save the model to the /models folder
save_path = ROOT / "models" / "final_sales_model.pkl"
joblib.dump(model, save_path)

print("\n--- Final Model Training Complete! ---")
print(f"Model saved to: {save_path}")