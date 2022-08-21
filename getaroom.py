# Import Libraries
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import make_pipeline
from tpot.builtins import StackingEstimator

# read in data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# drop columns that are not needed
train.drop(['Property_ID'], axis=1, inplace=True)
testPrpId = y = test['Property_ID']
test.drop(['Property_ID'], axis=1, inplace=True)

# split into X and y
y = train['Habitability_score']
train.drop(['Habitability_score'], axis=1, inplace=True)

# Converting string columns to numeric
categorical_features = ['Property_Type', 'Furnishing', 'Frequency_of_Powercuts', 'Power_Backup', 'Water_Supply', 'Crime_Rate', 'Dust_and_Noise']
labelencoder = LabelEncoder()
for feature in categorical_features:
    train[feature] = train[feature].astype(str)
    test[feature] = test[feature].astype(str)

    labelencoder.fit(train[feature])

    train[feature] = labelencoder.transform(train[feature])
    test[feature] = labelencoder.transform(test[feature])

    train[feature].fillna(train[feature].mean(), inplace = True)
    test[feature].fillna(train[feature].mean(), inplace = True)

# impute missing values
numerical_features = ['Property_Area', 'Number_of_Windows', 'Number_of_Doors','Traffic_Density_Score','Air_Quality_Index','Neighborhood_Review']
for feature in numerical_features:
    train[feature].fillna(train[feature].mean(), inplace = True)
    test[feature].fillna(test[feature].mean(), inplace = True)


# pipeline for categorical variables
categorical_features = ['Property_Type', 'Furnishing', 'Frequency_of_Powercuts', 'Power_Backup', 'Water_Supply', 'Crime_Rate', 'Dust_and_Noise']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# pipeline for numerical variables
numerical_features = ['Property_Area', 'Number_of_Windows', 'Number_of_Doors','Traffic_Density_Score','Air_Quality_Index','Neighborhood_Review']
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])

# transform full pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

preprocessor.fit(train)
train = preprocessor.transform(train)
test = preprocessor.transform(test)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train = sc.fit_transform(train)
test = sc.transform(test)


# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.2, random_state=42)


# Define models
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=ElasticNetCV(l1_ratio=0.2, tol=0.001)),
    RandomForestRegressor(bootstrap=True, max_features=0.7000000000000001,
                         min_samples_leaf=5, min_samples_split=13, n_estimators=100)
)

# Evaluate
exported_pipeline.fit(X_train, y_train)
y_pred = exported_pipeline.predict(X_test)
print("R-squared: %f" % r2_score(y_test, y_pred))
