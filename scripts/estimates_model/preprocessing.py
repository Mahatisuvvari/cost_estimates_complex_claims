import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from utils import Utils

utils = Utils()

class DataPreprocessor:
    def __init__(self, data):
        """
        Initializes the DataPreprocessor with the data.

        Parameters:
        data (DataFrame): The dataset to preprocess.
        """
        self.data = data
    
    def run_preprocessing(self):
        """
        Runs the preprocessing steps: handle missing values, feature engineering, split and scale data.

        Returns:
        X_train, X_test, y_train, y_test: Preprocessed and split data.
        """
        self.handle_missing_values()
        X, y = self.feature_engineering()

        X_train, X_test, y_train, y_test = utils.split_data(X,y)

        return X_train, X_test, y_train, y_test
    
    
    def handle_missing_values(self):
        """
        Handles missing values in the dataset by filling them with 0, if the column is Marital Status then assign 'U'.
        """

        # Check for missing values
        missing_values = self.data.isnull().sum()

        # Display columns with missing values
        missing_columns = missing_values[missing_values > 0]

        # Extract the column names with missing values
        missing_column_names = missing_columns.index.tolist()

        for column in missing_column_names:
            if column == 'MaritalStatus':
                self.data['MaritalStatus'].fillna('U', inplace=True)
            else:
                self.data.fillna(0, inplace=True)
    
    def feature_engineering(self):

        """
        Separates features and target variable and performs feature engineering.

        Returns:
        tuple: Features (X) and target variable (y).
        """

        # Convert datetime columns to datetime format
        self.data['DateTimeOfAccident'] = pd.to_datetime(self.data['DateTimeOfAccident'], errors='coerce')
        self.data['DateReported'] = pd.to_datetime(self.data['DateReported'], errors='coerce')

        # Calculate the time difference between DateTimeOfAccident and DateReported
        self.data['AccidentToReportDays'] = (self.data['DateReported'] - self.data['DateTimeOfAccident']).dt.days

        # Extract year, month, and day from the DateTimeOfAccident column
        self.data['AccidentYear'] = self.data['DateTimeOfAccident'].dt.year
        self.data['AccidentMonth'] = self.data['DateTimeOfAccident'].dt.month
        self.data['AccidentDay'] = self.data['DateTimeOfAccident'].dt.day

        # Extract year, month, and day from the DateReported column
        self.data['ReportYear'] = self.data['DateReported'].dt.year
        self.data['ReportMonth'] = self.data['DateReported'].dt.month
        self.data['ReportDay'] = self.data['DateReported'].dt.day

        # Drop time features
        self.data.drop(['DateTimeOfAccident', 'DateReported'], axis=1, inplace=True)

        # One hot Encoding
        self.one_hot_encoder()

        # Standard Scalar
        scaled_df = self.standard_scaler()

        # 'UltimateIncurredClaimCost' is the target variable
        X = scaled_df.drop(columns=['UltimateIncurredClaimCost'])
        y = scaled_df['UltimateIncurredClaimCost']

        return X,y

    def standard_scaler(self):
        """
        Applies standard scaling to continuous features and returns the scaled dataset.

        Returns:
        DataFrame: Scaled dataset.
        """
        # List of continuous features to scale
        continuous_features = [
            'DependentChildren', 'DependentsOther', 'HoursWorkedPerWeek',
            'DaysWorkedPerWeek', 'ClaimDescriptionKeyword_0',
            'ClaimDescriptionKeyword_1', 'ClaimDescriptionKeyword_2',
            'ClaimDescriptionKeyword_3', 'ClaimDescriptionKeyword_4',
            'ClaimDescriptionKeyword_5', 'ClaimDescriptionKeyword_6',
            'ClaimDescriptionKeyword_7', 'ClaimDescriptionKeyword_8',
            'ClaimDescriptionKeyword_9', 'ClaimDescriptionKeyword_10',
            'ClaimDescriptionKeyword_11', 'AccidentToReportDays', 'AccidentYear', 
            'AccidentMonth', 'AccidentDay', 'ReportYear', 'ReportMonth', 
            'ReportDay', 'Age', 'WeeklyWages', 'InitialIncurredClaimsCost'
        ]
        
        # Initialize the StandardScaler
        scaler = StandardScaler()
        
        # Fit and transform the continuous features
        self.data[continuous_features] = scaler.fit_transform(self.data[continuous_features])

        # Display the scaled continuous features
        scaled_df = self.data.copy()

        return scaled_df
     
    def one_hot_encoder(self):
         
        """
        Applies one-hot encoding to categorical features and updates the dataset.
        """
        # List of categorical features to encode
        categorical_features = ['Gender', 'MaritalStatus', 'PartTimeFullTime']
        
        # Initialize the OneHotEncoder
        encoder = OneHotEncoder(sparse=False)
        
        # Fit and transform the categorical features
        encoded_features = encoder.fit_transform(self.data[categorical_features])

        # Convert encoded features to DataFrame
        encoded_features_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))

        # Concatenate encoded features with the main dataset
        self.data = pd.concat([self.data, encoded_features_df], axis=1)

        # Drop original categorical columns
        self.data.drop(columns=categorical_features, inplace=True)

    

