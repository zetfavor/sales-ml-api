import pandas as pd
import optuna
import mlflow
import copy # We'll need this to copy our config dict
from pathlib import Path

# Import our custom source-code modules
from src.data_loader import load_config, load_raw
from src.model_train import train_model

def objective(trial: optuna.trial.Trial,
              base_config: dict,
              X_data: pd.DataFrame,
              y_data: pd.Series) -> float:
    """
    This is the main function Optuna will call for each "trial".
    A "trial" is one full experiment with one set of parameters.
    """

    # --- 1. Define the Search Space ---
    # We tell Optuna which parameters to "guess" and in what range.
    params_to_tune = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 600, step=50),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
    }

    # --- 2. Create the Config for this Trial ---
    # We make a deep copy of our base config...
    trial_config = copy.deepcopy(base_config)

    # ...and then UPDATE it with the new "guessed" parameters
    trial_config['training']['model']['params'].update(params_to_tune)

    # --- 3. Run and Log the Experiment with MLflow ---
    # We nest an MLflow run *inside* the Optuna trial
    with mlflow.start_run(nested=True):

        # Log all parameters: the base ones AND the ones we're tuning
        mlflow.log_params(trial_config['training'])
        mlflow.log_params(trial_config['training']['model']['params'])
        mlflow.set_tag("description", "Optuna tuning trial")

        # Train the model using our trusted src module
        # We use smote: true (from our config) for all tuning runs
        model, metrics = train_model(X_data, y_data, trial_config)

        # Log the results
        mlflow.log_metrics(metrics)

        # Get the score we want to maximize (our "objective")
        score = metrics['f1_class_1']
        mlflow.log_metric("tuning_objective_score", score)

    # --- 4. Return the Score to Optuna ---
    return score


# --- Main script execution ---
if __name__ == "__main__":

    # 1. Load our base config and data ONCE
    print("Loading base config and data...")
    config = load_config()
    df = load_raw("sample_sales.csv")
    X = df.drop("target", axis=1)
    y = df["target"]

    # --- Make sure we tune using the SMOTE config ---
    if not config['training']['smote']:
        print("WARNING: 'smote' is false in config.yaml. Forcing to 'true' for this tuning run.")
        config['training']['smote'] = True

    # 2. Set up the MLflow Experiment for this *entire tuning session*
    mlflow.set_experiment("Sales Tuning (Optuna)")

    # 3. Create the Optuna "Study"
    # We tell it we want to "maximize" our objective (the F1 score)
    study = optuna.create_study(direction="maximize")

    # 4. Run the optimization!
    # We use a lambda to pass our extra args (config, X, y) to the objective
    print("Starting Optuna tuning... This will take a few minutes.")
    study.optimize(
        lambda trial: objective(trial, base_config=config, X_data=X, y_data=y),
        n_trials=50  # Let's run 50 different experiments
    )

    # 5. Print the best results
    print("\n--- Tuning Complete! ---")
    print(f"Best trial number: {study.best_trial.number}")
    print(f"Best F1 Score (class 1): {study.best_value:.4f}")
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")