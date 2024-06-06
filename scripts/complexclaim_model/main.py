from preprocessing import DataPreprocessor
from model import ComplexClaimModel
import pandas as pd
from sklearn.model_selection import GridSearchCV
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
    4. Identify the best model based on recall for 'yes' class.
    """
    # Load data
    data_path = './data/data.csv'
    data = pd.read_csv(data_path)
    data.drop(['ClaimNumber'], axis=1, inplace=True)

    # You can select, which scenario of complex claims decision you want to model with.

    scenario = 'scenario_2'
    
    # Preprocess data
    preprocessor = DataPreprocessor(data, scenario)
    X_train, X_test, y_train, y_test, le = preprocessor.run_preprocessing()

    results = []
    # Train model
    for model_selector in ['logistic_regression', 
                           'random_forest',
                            'gradient_boosting',
                            'xgboost', 
                            ]:

        print("Running Model : {}".format(model_selector))

        model = ComplexClaimModel(model_type=model_selector)

        # Training model
        model.train(X_train, y_train)
    
        # Evaluate model
        test_accuracy, y_pred_decoded, y_test_decoded, train_accuracy, test_report = model.evaluate(X_train, y_train, X_test, y_test, le)

        # Plot metrics
        utils.evaluate_and_plot_model(model_selector, y_test_decoded, y_pred_decoded, train_accuracy, test_accuracy)
        utils.explain_classification_report(y_test_decoded, y_pred_decoded)

        # Append results to the DataFrame
        recall_yes = test_report['1']['recall']

        # Save results
        results.append({
            'Model': model_selector,
            'Test Accuracy': test_accuracy,
            'Train Accuracy': train_accuracy,
            'Recall (yes)': recall_yes,
            'Classification Report': test_report
        })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Find the best model based on recall for 'yes'
    best_model = results_df.loc[results_df['Recall (yes)'].idxmax()]
    print(f"Best model: {best_model['Model']} with Recall (yes): {best_model['Recall (yes)']}")

        
    # Define hyperparameter search space
    param_grid = {
        'random_forest': {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2']
        },
        'xgboost': {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [10, 20, 30, 40, 50],
            'learning_rate': [0.01, 0.1, 0.2, 0.3],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        },
        'gradient_boosting': {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2'],
            'learning_rate': [0.01, 0.1, 0.2, 0.3]
        }
    }

    # Picking the best model
    best_model_type = best_model['Model']
    # Initialize the best model with default parameters
    best_model = ComplexClaimModel(model_type=best_model_type)

    # Perform GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(
        estimator=best_model.model,
        param_grid=param_grid[best_model_type],
        scoring='recall',  # Adjust the scoring metric as needed
        cv=3,
        verbose=2,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    # Train the best model with optimized hyperparameters
    best_model.model = grid_search.best_estimator_
    best_model.train(X_train, y_train)

    # Evaluate the optimized model
    test_accuracy, y_pred_decoded, y_test_decoded, train_accuracy, test_report = model.evaluate(X_train, y_train, X_test, y_test, le)
    
    # Plot metrics
    utils.evaluate_and_plot_model(best_model_type, y_test_decoded, y_pred_decoded, train_accuracy, test_accuracy)
    utils.explain_classification_report(y_test_decoded, y_pred_decoded)


if __name__ == "__main__":
    main()
