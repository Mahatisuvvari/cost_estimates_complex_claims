from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from xgboost import XGBRegressor
from utils import Utils

utils = Utils()

class ClaimCostModel:
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
            return RandomForestRegressor(n_estimators=300, random_state=42)
        elif model_type == 'xgboost':
            return XGBRegressor(n_estimators=300,  # Number of boosting rounds
                                learning_rate=0.1,  # Step size shrinkage used to prevent overfitting
                                max_depth=3,  # Maximum depth of a tree
                                min_child_weight=1,  # Minimum sum of instance weight (hessian) needed in a child
                                subsample=0.8,  # Subsample ratio of the training instances
                                colsample_bytree=0.8,  # Subsample ratio of columns when constructing each tree
                                objective='reg:absoluteerror',  # Specify the learning task and the corresponding learning objective
                                n_jobs=-1,  # Number of parallel threads used to run xgboost
                                random_state=42)
        elif model_type == 'linear_regression':
            return LinearRegression()
        elif model_type == 'gradient_boosting':
            return GradientBoostingRegressor(n_estimators=300,  # Number of boosting stages to be run
                                            learning_rate=0.1,  # Learning rate or shrinkage
                                            max_depth=3,  # Maximum depth of the individual trees
                                            min_samples_split=2,  # Minimum number of samples required to split an internal node
                                            min_samples_leaf=1,  # Minimum number of samples required to be at a leaf node
                                            subsample=0.8,  # Fraction of samples to be used for fitting the individual base learners
                                            loss='absolute_error',
                                            random_state=42)
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
    
    def evaluate(self, X_test, y_test):
        """
        Evaluates the model on the test data and prints the Mean Squared Error.

        Parameters:
        X_test (array-like): Test features.
        y_test (array-like): Test target.
        
        Returns:
        tuple: Mean Absolute Error (MAE), R-squared (R2), and Mean Absolute Percentage Error (MAPE).
        """
        model = utils.load_model( self.model_type)
        predictions = model.predict(X_test)

        # Evaluate the model
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        mape = mean_absolute_percentage_error(y_test, predictions)

        print(f"Regressor MAE: {mae}")
        print(f"Regressor R2: {r2}")
        print(f"Regressor MAPE: {mape}")

        return mae, r2, mape

    def predict(self, X):
        """
        Makes predictions using the trained model.

        Parameters:
        X (array-like): Features to predict.

        Returns:
        predictions (array-like): Model predictions.
        """
        model = utils.load_model(f'{self.model_type}_model.pkl')
        return model.predict(X)
