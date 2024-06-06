from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from collections import Counter
import os


class Utils:

    def create_folder_if_not_exists(self, final_path):
        """
        Create a folder if it doesn't exist and return the folder path.

        Parameters:
        final_path (str): The path of the folder to create.

        Returns:
        str: Absolute path of the created folder.
        """
        
        if not os.path.exists(final_path):
            os.makedirs(final_path)
        
        return os.path.abspath(final_path)

    def split_data(self, X, y, test_size=0.3, random_state=42):
        """
        Splits the data into training and testing sets.

        Parameters:
        X (DataFrame): Features.
        y (Series): Target variable.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Controls the shuffling applied to the data before applying the split.

        Returns:
        X_train, X_test, y_train, y_test: Split data.
        """
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def save_model(self, model, folder_name):
        """
        Saves the trained model to a file.

        Parameters:
        model: Trained model.
        folder_name (str): Name of the folder to save the model in.
        """
        final_path = os.path.join( 'models', 'complexclaim_model', folder_name)
        folderpath = self.create_folder_if_not_exists(final_path)
        filepath = os.path.join(folderpath, 'model.pkl')

        joblib.dump(model, filepath)

    def load_model(self, folder_name):
        """
        Loads a model from a file.

        Parameters:
        folder_name (str): Name of the folder to load the model from.

        Returns:
        model: Loaded model.
        """

        final_path = os.path.join( 'models', 'complexclaim_model', folder_name)
        folderpath = self.create_folder_if_not_exists(final_path)
        filepath = os.path.join(folderpath, 'model.pkl')
        return joblib.load(filepath)
    
    def explain_classification_report(self, y_true, y_pred):
        """
        Prints an explanation of the classification report metrics.

        Parameters:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        """
                
        report = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        
        print("Classification Report Explanation:\n")
        
        for idx, row in report_df.iterrows():
            if idx in ['accuracy', 'macro avg', 'weighted avg']:
                continue  # Skip overall accuracy and avg rows
            
            print(f"Class: {idx}")
            print(f"  Precision: {row['precision']:.2f}")
            print(f"  Recall: {row['recall']:.2f}")
            print(f"  F1-score: {row['f1-score']:.2f}")
            print(f"  Support: {int(row['support'])}")
            print("\nExplanation:")
            print(f"  - Precision: Out of all the predicted instances of class '{idx}', {row['precision']*100:.2f}% were correct.")
            print(f"  - Recall: Out of all the actual instances of class '{idx}', {row['recall']*100:.2f}% were correctly identified.")
            print(f"  - F1-score: The harmonic mean of precision and recall for class '{idx}'.")
            print(f"  - Support: The number of actual occurrences of class '{idx}' in the dataset.")
            print("\n")


    def evaluate_and_plot_model(self, model_name, y_test_decoded, y_pred_decoded, train_accuracy, test_accuracy):
        """
        Evaluates the model, plots the confusion matrix, and checks for overfitting.

        Parameters:
        model_name (str): Name of the model.
        y_test_decoded (array-like): Decoded true labels.
        y_pred_decoded (array-like): Decoded predicted labels.
        train_accuracy (float): Training accuracy.
        test_accuracy (float): Testing accuracy.
        """
        # Print the evaluation metrics
        print(f"{model_name}:")
        print(classification_report(y_test_decoded, y_pred_decoded))
        print(f"Accuracy: {accuracy_score(y_test_decoded, y_pred_decoded)}\n")

        # Compute confusion matrix
        cm = confusion_matrix(y_test_decoded, y_pred_decoded)

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['no', 'yes'], yticklabels=['no', 'yes'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {model_name}')
        fig_path = os.path.join('figures', 'complex_claim_model', model_name)
        dir = self.create_folder_if_not_exists(fig_path)
        fig_name = f'confusion_matrix_{model_name}.png'
        plt.savefig(os.path.join(dir,fig_name))
        plt.close()

        # Check for overfitting
        if train_accuracy > test_accuracy + 0.1:
            print("The model is likely overfitting. Training accuracy is significantly higher than testing accuracy.")
        else:
            print("The model is not overfitting. Training and testing accuracies are similar.")

    
        # Create a function for SMOTE resampling
    def apply_smote(self, X_train, y_train):
        """
        Applies SMOTE (Synthetic Minority Over-sampling Technique) to balance the classes.

        Parameters:
        X_train (array-like): Training features.
        y_train (array-like): Training target.

        Returns:
        tuple: Resampled training features and target (X_train_resampled, y_train_resampled).
        """

        # Display the class distribution before applying SMOTE
        print(f"Class distribution before SMOTE: {Counter(y_train)}")

        # Apply SMOTE to the entire dataset
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        # Display the class distribution after applying SMOTE
        print(f"Class distribution after SMOTE: {Counter(y_train_resampled)}")

        return X_train_resampled, y_train_resampled 
    
    def identify_and_drop_correlated_features(self, df):
        """
        Identifies and drops highly correlated features from the dataset.

        Parameters:
        df (DataFrame): The dataset.

        Returns:
        DataFrame: Reduced dataset with highly correlated features removed.
        """
        # Calculate the correlation matrix
        correlation_matrix = df[['Age', 'WeeklyWages', 'HoursWorkedPerWeek', 'DaysWorkedPerWeek',
            'InitialIncurredClaimsCost', 'UltimateIncurredClaimCost',
            'AccidentToReportDays', 'AccidentYear', 'AccidentMonth', 'AccidentDay',
            'ReportYear', 'ReportMonth', 'ReportDay', 'TotalDependents']].corr().abs()

        # Create a mask to ignore the upper triangle of the correlation matrix
        upper_triangle_mask = np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        upper_triangle = correlation_matrix.where(upper_triangle_mask)

        # Identify features with correlation greater than 0.8
        correlated_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.8)]

        # Drop the highly correlated features
        df_reduced = df.drop(columns=correlated_features)

        return df_reduced

