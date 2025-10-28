import pandas as pd
from sklearn.datasets import make_classification
from pathlib import Path

# Create a synthetic, imbalanced dataset
# 95% of samples are class 0, 5% are class 1
X, y = make_classification(
    n_samples=2000,
    n_features=15,
    n_informative=5,
    n_redundant=2,
    n_classes=2,
    weights=[0.95, 0.05], # This makes it imbalanced
    flip_y=0.01,
    random_state=42
)

# Convert to a DataFrame
feature_names = [f'feature_{i}' for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# Define save path using our project structure
save_path = Path(__file__).parent / "data" / "raw" / "sample_sales.csv"

# Save to the data/raw folder
df.to_csv(save_path, index=False)

print(f"Successfully created imbalanced dataset at:")
print(save_path)
print("\nData preview:")
print(df.head())
print(f"\nTarget distribution:\n{df['target'].value_counts(normalize=True)}")