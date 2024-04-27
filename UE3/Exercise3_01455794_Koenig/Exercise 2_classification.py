import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

data = pd.read_csv('merged_data.csv')

data['is_delayed'] = (data['ARR_DELAY'] >= 30).astype(int)

print(data.columns)


data = data.replace('M', np.nan)
missing_values_count = data.isnull().sum()
missing_percentage = (missing_values_count / len(data)) * 100
print("Missing Percentage before cleaning")
print(missing_percentage)
columns_to_drop = missing_percentage[missing_percentage > 30].index
data = data.drop(columns=columns_to_drop)

print(data.head(10))


numerical_features = [
    'CRS_DEP_TIME', 'DEP_TIME', 'TAXI_OUT', 'DISTANCE',
    'tmpf_dep', 'dwpf_dep', 'relh_dep', 'drct_dep', 'sknt_dep',
    'p01i_dep', 'alti_dep', 'vsby_dep', 'feel_dep', 'skyl1_dep'
]

categorical_features = [
    'OP_UNIQUE_CARRIER', 'TAIL_NUM', 'OP_CARRIER_FL_NUM',
    'ORIGIN', 'DEST', 'skyc1_dep'
]

# Convert columns that should be numeric but are object type due to mixed data
for col in numerical_features:
    # Convert all entries to numeric, setting errors='coerce' will convert non-convertible values to NaN
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Filled with NaN values
data[numerical_features] = data[numerical_features].fillna(method='ffill').fillna(method='bfill')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),  # Scale numerical features
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)  # One-hot encode categorical features
    ])

X = data[numerical_features + categorical_features]
X_processed = preprocessor.fit_transform(X)
y = data['is_delayed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Pipeline with preprocessing and decision tree classifier
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(criterion='entropy'))
])

# Train model
pipeline.fit(X_train, y_train)

# Extract and map feature importances
importances = pipeline.named_steps['classifier'].feature_importances_
feature_names = preprocessor.get_feature_names_out()

#help of ChatGPT
# Map aggregated importances
feature_importance_dict = {}
for feature, imp in zip(feature_names, importances):
    # Split the feature name on the last '_' and get the base feature
    base_feature = feature.rsplit('_', 1)[0]
    # If the base feature is already in the dictionary, add the importance
    if base_feature in feature_importance_dict:
        feature_importance_dict[base_feature] += imp
    # If the base feature is not in the dictionary, initialize it with the importance
    else:
        feature_importance_dict[base_feature] = imp

# Sort feature importances
sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
for feat, imp in sorted_features:
    print(f"{feat}: {imp:.4f}")

#take all features and calculate the prediction via decision tree classification

# Preprocess the selected features
preprocessor_selected = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='drop'  # This drops the columns that we do not transform
)

# Prepare the feature matrix
X_processed_selected = preprocessor_selected.fit_transform(X)

X_train_selected, X_test_selected, y_train, y_test = train_test_split(
    X_processed_selected, y, test_size=0.20, random_state=42
)

# Create a Decision Tree classifier instance
tree_model_selected = DecisionTreeClassifier(criterion='entropy', random_state=42)

# Train the model
tree_model_selected.fit(X_train_selected, y_train)

#evaluate the model on the test set
y_pred = tree_model_selected.predict(X_test_selected)
print("Accuracy on test data with all features:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


#take the best features (over 0.08) and make a decision tree classification

best_numerical_features = [
    'CRS_DEP_TIME', 'DEP_TIME', 'TAXI_OUT','sknt_dep', 'skyl1_dep'
]

best_categorical_features = [
    'OP_UNIQUE_CARRIER', 'TAIL_NUM', 'OP_CARRIER_FL_NUM',
    'ORIGIN', 'DEST', 'skyc1_dep'
]

top_numerical_features = [
    'CRS_DEP_TIME', 'DEP_TIME', 'TAXI_OUT'
]

top_categorical_features = [
    'OP_UNIQUE_CARRIER', 'TAIL_NUM'
]

worst_numerical_features = [
    'tmpf_dep', 'dwpf_dep', 'relh_dep', 'drct_dep',
    'p01i_dep', 'alti_dep', 'vsby_dep', 'feel_dep'
]

worst_categorical_features = [
    'OP_UNIQUE_CARRIER', 'skyc1_dep'
]


X_best = data[best_numerical_features + best_categorical_features]

# Preprocess the selected features
preprocessor_selected = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), best_numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), best_categorical_features)
    ],
    remainder='drop'  # This drops the columns that we do not transform
)

# Prepare the feature matrix
X_processed_selected = preprocessor_selected.fit_transform(X_best)

X_train_selected, X_test_selected, y_train, y_test = train_test_split(
    X_processed_selected, y, test_size=0.20, random_state=42
)

# Create a Decision Tree classifier instance
tree_model_selected = DecisionTreeClassifier(criterion='entropy', random_state=42)

# Train the model
tree_model_selected.fit(X_train_selected, y_train)

# evaluate the model on the test set


y_pred = tree_model_selected.predict(X_test_selected)
print("Accuracy on test data with the features with a information gain over 0.08:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

#take only the top 5 features and make a decision tree classification
X_top = data[top_numerical_features + top_categorical_features]

# Preprocess the selected features
preprocessor_selected = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), top_numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), top_categorical_features)
    ],
    remainder='drop'  # This drops the columns that we do not transform
)


# Prepare the feature matrix
X_processed_selected = preprocessor_selected.fit_transform(X_top)

X_train_selected, X_test_selected, y_train, y_test = train_test_split(
    X_processed_selected, y, test_size=0.20, random_state=42
)

# Create a Decision Tree classifier instance
tree_model_selected = DecisionTreeClassifier(criterion='entropy', random_state=42)

# Train the model
tree_model_selected.fit(X_train_selected, y_train)

#evaluate the model on the test set

y_pred = tree_model_selected.predict(X_test_selected)
print("Accuracy on test data with all features that have the top 5 information gains:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


#take only the top 5 features and make a decision tree classification

top_numerical_features = [
    'CRS_DEP_TIME', 'DEP_TIME', 'TAXI_OUT'
]

top_categorical_features = [
    'OP_UNIQUE_CARRIER', 'TAIL_NUM'
]




#take only the worst 10 features and make a decision tree classification

X_worst = data[worst_numerical_features + worst_categorical_features]

# Preprocess the selected features
preprocessor_selected = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), worst_numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), worst_categorical_features)
    ],
    remainder='drop'  # This drops the columns that we do not transform
)

# Prepare the feature matrix
X_processed_selected = preprocessor_selected.fit_transform(X_worst)

X_train_selected, X_test_selected, y_train, y_test = train_test_split(
    X_processed_selected, y, test_size=0.20, random_state=42
)

# Create a Decision Tree classifier instance
tree_model_selected = DecisionTreeClassifier(criterion='entropy', random_state=42)

# Train the model
tree_model_selected.fit(X_train_selected, y_train)

#evaluate the model on the test set

y_pred = tree_model_selected.predict(X_test_selected)
print("Accuracy on test data with the worst 5 features:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
