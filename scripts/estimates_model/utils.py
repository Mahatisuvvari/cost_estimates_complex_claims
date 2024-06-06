from sklearn.model_selection import train_test_split
import joblib
import numpy as np
import matplotlib.pyplot as plt
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

    def split_data(self, X, y, test_size=0.2, random_state=42):
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
        final_path = os.path.join( 'models', 'estimates_model', folder_name)
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
        final_path = os.path.join( 'models', 'estimates_model', folder_name)    
        folderpath = self.create_folder_if_not_exists(final_path)
        filepath = os.path.join(folderpath, 'model.pkl')
        return joblib.load(filepath)
    
    def plot_feature_importances(self, model, feature_names, model_name):
        """
        Plots the feature importances and saves the plot to a file.

        Parameters:
        model: Trained model.
        feature_names (list): List of feature names.
        model_name (str): Name of the model for saving the plot.
        """
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)

            plt.figure(figsize=(10, 8))
            plt.title(f'Feature Importances for {model_name}')
            plt.barh(range(len(indices)), importances[indices], align='center')
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel('Relative Importance')
            fig_path = os.path.join('figures', 'estimates_model', model_name)
            dir = self.create_folder_if_not_exists(fig_path)
            fig_name = f'feature_importances_{model_name}.png'
            plt.savefig(os.path.join(dir,fig_name))
            plt.close()

