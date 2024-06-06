import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from utils import Utils

utils = Utils()

class DataPreprocessor:
    def __init__(self, data, scenario):
        """
        Initializes the DataPreprocessor with the data.

        Parameters:
        data (DataFrame): The dataset to preprocess.
        """
        self.data = data
        self.scenario = scenario
    
    def run_preprocessing(self):
        """
        Runs the preprocessing steps: handle missing values, feature engineering, split and scale data.

        Returns:
        tuple: Preprocessed and split data (X_train, X_test, y_train, y_test, le).
        """
        self.handle_missing_values()
        X, y, le = self.feature_engineering(self.scenario)

        X_train, X_test, y_train, y_test = utils.split_data(X,y, test_size=0.3)

        X_train_resampled, y_train_resampled = utils.apply_smote(X_train, y_train)

        return X_train_resampled, X_test, y_train_resampled, y_test, le
    
    
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

    def complex_claim_scenario_1(self):
        """
        Creates a composite score for complex claims based on multiple criteria and creates a binary label.
        """

        # Define thresholds and calculate scores for each criterion
        high_initial_threshold = self.data['InitialIncurredClaimsCost'].quantile(0.50)

        low_initial_high_ultimate = ((self.data['InitialIncurredClaimsCost'] < self.data['InitialIncurredClaimsCost'].quantile(0.50)) &
                                    (self.data['UltimateIncurredClaimCost'] > high_initial_threshold))

        age_with_dependents = ((self.data['Age'] > 45) & (self.data['TotalDependents'] > 0))

        low_wages_high_claims = ((self.data['WeeklyWages'] < self.data['WeeklyWages'].quantile(0.20)) &
                                (self.data['InitialIncurredClaimsCost'] > high_initial_threshold))

        low_hours_days_high_claims = ((self.data['HoursWorkedPerWeek'] < self.data['HoursWorkedPerWeek'].quantile(0.20)) |
                                    (self.data['DaysWorkedPerWeek'] < self.data['DaysWorkedPerWeek'].quantile(0.20)) &
                                    (self.data['InitialIncurredClaimsCost'] > high_initial_threshold))

        high_days_report_high_claim = ((self.data['AccidentToReportDays'] > self.data['AccidentToReportDays'].quantile(0.50)) &
                                    (self.data['InitialIncurredClaimsCost'] > high_initial_threshold))

        # Create scores for each criterion
        self.data['Score_HighInitial'] = (self.data['InitialIncurredClaimsCost'] > high_initial_threshold).astype(int)
        self.data['Score_LowInitialHighUltimate'] = low_initial_high_ultimate.astype(int)
        self.data['Score_AgeWithDependents'] = age_with_dependents.astype(int)
        self.data['Score_LowWagesHighClaims'] = low_wages_high_claims.astype(int)
        self.data['Score_LowHoursDaysHighClaims'] = low_hours_days_high_claims.astype(int)
        self.data['Score_HighDaysReportHighClaim'] = high_days_report_high_claim.astype(int)


        # Define weights for each criterion
        weights = {
            'Score_HighInitial': 0.3,
            'Score_LowInitialHighUltimate': 0.2,
            'Score_AgeWithDependents': 0.15,
            'Score_LowWagesHighClaims': 0.15,
            'Score_LowHoursDaysHighClaims': 0.1,
            'Score_HighDaysReportHighClaim': 0.1
        }

        # Calculate composite score
        self.data['CompositeScore'] = (self.data['Score_HighInitial'] * weights['Score_HighInitial'] +
                                self.data['Score_LowInitialHighUltimate'] * weights['Score_LowInitialHighUltimate'] +
                                self.data['Score_AgeWithDependents'] * weights['Score_AgeWithDependents'] +
                                self.data['Score_LowWagesHighClaims'] * weights['Score_LowWagesHighClaims'] +
                                self.data['Score_LowHoursDaysHighClaims'] * weights['Score_LowHoursDaysHighClaims'] +
                                self.data['Score_HighDaysReportHighClaim'] * weights['Score_HighDaysReportHighClaim'])

        self.data = self.data.loc[:, ~self.data.columns.str.startswith('Score_')]

        # Define threshold for complex claims
        threshold = 0.4

        # Create binary label based on the composite score
        self.data['ComplexClaim'] = self.data['CompositeScore'] > threshold
        self.data['ComplexClaim'] = self.data['ComplexClaim'].map({True: 'Yes', False: 'No'})

        # Dropping Composite Score
        self.data.drop('CompositeScore', axis=1, inplace=True)
        
            
    def feature_engineering(self, scenario):
        """
        Separates features and target variable and performs feature engineering.

        Returns:
        tuple: Features (X), target variable (y), and label encoder (le).
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

        # Create Total Dependents feature
        self.data['TotalDependents'] = self.data['DependentChildren'] + self.data['DependentsOther']

        # Drop time features
        self.data.drop(['DateTimeOfAccident', 'DateReported'], axis=1, inplace=True)

        # Define and create complex claim field
        if scenario == 'scenario_1':
            self.complex_claim_scenario_1()
        else:
            self.complex_claim_scenario_2()

        # One hot Encoding
        self.one_hot_encoder()

        # Standard Scalar
        scaled_df = self.standard_scaler()

        # Dropping outliers
        df_reduced = utils.identify_and_drop_correlated_features(scaled_df)

        # 'UltimateIncurredClaimCost' is dropped as it is a derivative
        df_reduced.drop(['UltimateIncurredClaimCost', 'DependentChildren', 'DependentsOther'], axis=1, inplace=True)

        # Target = 'ComplexClaim'
        X = df_reduced.drop(columns=['ComplexClaim'])
        y = df_reduced['ComplexClaim']

        y_encoded, le = self.label_encoder(y)

        return X,y_encoded, le
    
    def label_encoder(self, y):
        """
        Encodes the target variable using LabelEncoder.

        Parameters:
        y (Series): Target variable.

        Returns:
        tuple: Encoded target variable and label encoder.
        """
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        return y_encoded, le

    def standard_scaler(self):
        """
        Applies standard scaling to continuous features and returns the scaled dataset.

        The function scales the continuous features using StandardScaler, which standardizes
        the features by removing the mean and scaling to unit variance. It updates the dataset
        with the scaled values and returns a copy of the scaled dataset.

        Returns:
        DataFrame: Scaled dataset.
        """
        # List of continuous features to scale
        continuous_features = [
            'TotalDependents', 'HoursWorkedPerWeek',
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

        The function encodes the categorical features 'Gender', 'MaritalStatus', 
        and 'PartTimeFullTime' using one-hot encoding, concatenates the encoded 
        features with the original dataset, and drops the original categorical columns.
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

    def complex_claim_scenario_2(self):
        """
        Creates a composite score for complex claims based on multiple criteria and creates a binary label.
        """

        # Define thresholds and calculate scores for each criterion
        high_ultimate_threshold = self.data['UltimateIncurredClaimCost'].quantile(0.50)

        low_initial_high_ultimate = ((self.data['InitialIncurredClaimsCost'] < self.data['InitialIncurredClaimsCost'].quantile(0.50)) &
                                    (self.data['UltimateIncurredClaimCost'] > high_ultimate_threshold))

        age_with_dependents = ((self.data['Age'] > 45) & (self.data['TotalDependents'] > 0))

        low_wages_high_claims = ((self.data['WeeklyWages'] < self.data['WeeklyWages'].quantile(0.20)) &
                                (self.data['UltimateIncurredClaimCost'] > high_ultimate_threshold))

        low_hours_days_high_claims = ((self.data['HoursWorkedPerWeek'] < self.data['HoursWorkedPerWeek'].quantile(0.20)) |
                                    (self.data['DaysWorkedPerWeek'] < self.data['DaysWorkedPerWeek'].quantile(0.20)) &
                                    (self.data['UltimateIncurredClaimCost'] > high_ultimate_threshold))

        high_days_report_high_claim = ((self.data['AccidentToReportDays'] > self.data['AccidentToReportDays'].quantile(0.50)) &
                                    (self.data['UltimateIncurredClaimCost'] > high_ultimate_threshold))

        # Create scores for each criterion
        self.data['Score_HighUltimate'] = (self.data['UltimateIncurredClaimCost'] > high_ultimate_threshold).astype(int)
        self.data['Score_LowInitialHighUltimate'] = low_initial_high_ultimate.astype(int)
        self.data['Score_AgeWithDependents'] = age_with_dependents.astype(int)
        self.data['Score_LowWagesHighClaims'] = low_wages_high_claims.astype(int)
        self.data['Score_LowHoursDaysHighClaims'] = low_hours_days_high_claims.astype(int)
        self.data['Score_HighDaysReportHighClaim'] = high_days_report_high_claim.astype(int)

        # Define weights for each criterion
        weights = {
            'Score_HighUltimate': 0.3,
            'Score_LowInitialHighUltimate': 0.2,
            'Score_AgeWithDependents': 0.15,
            'Score_LowWagesHighClaims': 0.15,
            'Score_LowHoursDaysHighClaims': 0.1,
            'Score_HighDaysReportHighClaim': 0.1
        }

        # Calculate composite score
        self.data['CompositeScore'] = (self.data['Score_HighUltimate'] * weights['Score_HighUltimate'] +
                                self.data['Score_LowInitialHighUltimate'] * weights['Score_LowInitialHighUltimate'] +
                                self.data['Score_AgeWithDependents'] * weights['Score_AgeWithDependents'] +
                                self.data['Score_LowWagesHighClaims'] * weights['Score_LowWagesHighClaims'] +
                                self.data['Score_LowHoursDaysHighClaims'] * weights['Score_LowHoursDaysHighClaims'] +
                                self.data['Score_HighDaysReportHighClaim'] * weights['Score_HighDaysReportHighClaim'])

        self.data = self.data.loc[:, ~self.data.columns.str.startswith('Score_')]

        # Define threshold for complex claims
        threshold = 0.4

        # Create binary label based on the composite score
        self.data['ComplexClaim'] = self.data['CompositeScore'] > threshold
        self.data['ComplexClaim'] = self.data['ComplexClaim'].map({True: 'Yes', False: 'No'})

        # Dropping Composite Score
        self.data.drop('CompositeScore', axis=1, inplace=True)

    

