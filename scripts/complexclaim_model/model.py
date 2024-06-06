from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from utils import Utils

utils = Utils()

class ComplexClaimModel:
    def __init__(self, model_type='random_forest'):
        """
        Initializes the ClaimCostModel with the selected model type.

        Parameters:
        model_type (str): The type of model to use ('random_forest', 'xgboost', 'linear_regression', 'gradient_boosting').
        """
        self.model_type = model_type
        self.model = self._initialize_model(model_type)
    
    def _initialize_model(self, model_type):
        """
        Initializes the model based on the selected model type.

        Parameters:
        model_type (str): The type of model to use.

        Returns:
        model: Initialized model.
        """
        if model_type == 'random_forest':
            return RandomForestClassifier(random_state=42)
        elif model_type == 'xgboost':
            return XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        elif model_type == 'logistic_regression':
            return LogisticRegression(random_state=42)
        elif model_type == 'gradient_boosting':
            return GradientBoostingClassifier(random_state=42)
        else:
            raise ValueError("Unsupported model type. Choose from 'random_forest', 'xgboost', 'linear_regression', 'gradient_boosting'.")

    def train(self, X_train, y_train):
        """
        Trains the selected model and saves it to a file.

        Parameters:
        X_train (array-like): Training features.
        y_train (array-like): Training target.
        """
        self.model.fit(X_train, y_train)
        utils.save_model(self.model, self.model_type)
    
    def evaluate(self, X_train, y_train, X_test, y_test, le):
        """
        Evaluates the model on the test data and prints the accuracy and classification report.

        Parameters:
        X_train (array-like): Training features.
        y_train (array-like): Training target.
        X_test (array-like): Test features.
        y_test (array-like): Test target.
        le (LabelEncoder): Label encoder used for decoding the target variable.

        Returns:
        tuple: Test accuracy, decoded predictions, decoded test targets, train accuracy, and classification report.
        """
        model = utils.load_model( self.model_type)
        predictions = model.predict(X_test)

        # Test Accuracy
        test_accuracy = accuracy_score(y_test, predictions)
        test_report = classification_report(y_test, predictions, output_dict=True)

        # Decode the predictions
        y_pred_decoded = le.inverse_transform(predictions)
        y_test_decoded = le.inverse_transform(y_test)

        # Evaluate on the training set
        y_train_pred = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        

        return test_accuracy, y_pred_decoded, y_test_decoded, train_accuracy, test_report

    def predict(self, X):
        """
        Makes predictions using the trained model.

        Parameters:
        X (array-like): Features to predict.

        Returns:
        array-like: Model predictions.
        """
        model = utils.load_model(f'{self.model_type}_model.pkl')
        return model.predict(X)
