from preprocessing import DataPreprocessor
from model import ClaimCostModel
import pandas as pd
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import matplotlib.pyplot as plt
import seaborn as sns
import os
from utils import Utils

utils = Utils()

def main():
    
    """
    Main function to load data, preprocess it, train the model, and evaluate the model.

    Steps:
    1. Load data from CSV file.
    2. Preprocess the data.
    3. Train multiple models and evaluate their performance.
    4. Perform hyperparameter tuning on the best model using Bayesian optimization.
    5. Evaluate the optimized model.
    """
    # Load data
    data_path = './data/data.csv'
    data = pd.read_csv(data_path)
    data.drop(['ClaimNumber'], axis=1, inplace=True)
    
    # Preprocess data
    preprocessor = DataPreprocessor(data)
    X_train, X_test, y_train, y_test = preprocessor.run_preprocessing()
    feature_names = X_train.columns

    # Results DataFrame
    results = pd.DataFrame(columns=['model', 'mae', 'r2', 'mape'])
    
    # Train model
    for model_selector in [
                            'linear_regression', 
                            'xgboost', 
                            'gradient_boosting',
                            'random_forest',
                            ]:

        print("Running Model : {}".format(model_selector))

        model = ClaimCostModel(model_type=model_selector)

        # Training model
        model.train(X_train, y_train)
    
        # Evaluate model
        mae, r2, mape = model.evaluate(X_test, y_test)

        # Plot and save feature importances
        plot_file = f'feature_importances_{model_selector}.png'
        utils.plot_feature_importances(model.model, feature_names, model_selector)

        # Append results to the DataFrame
        result = pd.DataFrame({'model': [model_selector], 'mae': [mae], 'r2': [r2], 'mape': [mape]})
        results = pd.concat([results, result], ignore_index=True)

        # Print all results
        print(results)

    # Select the best model based on MAE
    best_model_row = results.loc[results['mae'].idxmin()]
    best_model_type = best_model_row['model']

    print(f"Best model based on MAE: {best_model_type}")

    # Define hyperparameter search space
    search_spaces = {
        'random_forest': {
            'n_estimators': Integer(100, 500),
            'max_depth': Integer(10, 50),
            'min_samples_split': Integer(2, 10),
            'min_samples_leaf': Integer(1, 4),
            'max_features': Categorical(['auto', 'sqrt', 'log2'])
        },
        'xgboost': {
            'n_estimators': Integer(100, 500),
            'max_depth': Integer(10, 50),
            'learning_rate': Real(0.01, 0.3, prior='log-uniform')
        },
        'gradient_boosting': {
            'n_estimators': Integer(100, 500),
            'max_depth': Integer(10, 50),
            'min_samples_split': Integer(2, 10),
            'min_samples_leaf': Integer(1, 4),
            'max_features': Categorical(['auto', 'sqrt', 'log2'])
        }
    }

    # Initialize the best model with default parameters
    best_model = ClaimCostModel(model_type=best_model_type)

    # Perform BayesSearchCV for hyperparameter tuning
    optimizer = BayesSearchCV(
        estimator=best_model.model,
        search_spaces=search_spaces[best_model_type],
        n_iter=32,
        cv=3,
        scoring='neg_mean_absolute_error',
        random_state=42,
        verbose=2
    )

    optimizer.fit(X_train, y_train)

    # Train the best model with optimized hyperparameters
    best_model.model = optimizer.best_estimator_
    best_model.train(X_train, y_train)

    # Evaluate the optimized model
    mae, r2, mape = best_model.evaluate(X_test, y_test)
    print(f"Optimized Model MAE: {mae}, R2: {r2}, MAPE: {mape}")

    # Plot and save feature importances for the optimized model
    utils.plot_feature_importances(best_model.model, feature_names, f'{best_model_type}_optimized')

if __name__ == "__main__":
    main()
