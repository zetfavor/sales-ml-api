from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score
import pandas as pd

def train_model(X: pd.DataFrame, y: pd.Series, config: dict):
    """Trains the model with SMOTE based on config."""

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config['training']['test_size'],
        random_state=config['training']['random_seed']
    )

    if config['training']['smote']:
        print("Applying SMOTE...")
        smote = SMOTE(random_state=config['training']['random_seed'])
        X_train, y_train = smote.fit_resample(X_train, y_train)

    print("Training XGBoost model...")
    model = XGBClassifier(
        random_state=config['training']['random_seed'],
        **config['training']['model']['params']
    )

    model.fit(X_train, y_train)

    print("Evaluating model on test set...")
    preds = model.predict(X_test)

    # Print the full report for visual inspection
    print(classification_report(y_test, preds))

    # --- NEW PART ---
    # Calculate metrics to return for logging
    # We focus on the minority class (1)
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "f1_macro": f1_score(y_test, preds, average='macro'),
        "precision_class_1": precision_score(y_test, preds, pos_label=1),
        "recall_class_1": recall_score(y_test, preds, pos_label=1),
        "f1_class_1": f1_score(y_test, preds, pos_label=1)
    }

    return model, metrics