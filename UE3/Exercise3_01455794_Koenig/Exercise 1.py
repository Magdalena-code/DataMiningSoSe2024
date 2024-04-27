import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler




data = pd.read_csv("labor-negotiations_students.csv",  delimiter=";")

#print(data.columns.values)
#print(data.head(10))

#data cleaning
missing_values_count = data.isnull().sum()
missing_percentage = (missing_values_count / len(data)) * 100
print("Missing Percentage before cleaning")
print(missing_percentage)

#columns that have more then 30% values missing will be deleted immediately

columns_to_drop = missing_percentage[missing_percentage > 30].index
data = data.drop(columns=columns_to_drop)

missing_values_count = data.isnull().sum()
missing_percentage = (missing_values_count / len(data)) * 100
print("Missing Percentage after cleaning")
print(missing_percentage)

for column in data.columns:
    if pd.api.types.is_numeric_dtype(data[column]):
        median = data[column].median()
        data.fillna({column: median}, inplace=True)
    else:
        most_common = data[column].mode()[0]
        data.fillna({column: most_common}, inplace=True)

print(data.head(10))

def check_column_values(df):
    # Check continuous variables (assume they should be numeric types)
    continuous_columns = ['duration', 'wage1', 'wage2', 'hours', 'holidays']
    for column in continuous_columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            print(f"Column {column} should be numeric but has type {df[column].dtype}.")

    # Check categorical variables with specific expected categories
    vacation_categories = ['below average', 'average', 'generous']
    # Drop columns that have an unexpected value.
    if not set(df['vacation'].dropna().unique()).issubset(vacation_categories):
        print(f"Column 'vacation' had unexpected values.")

    # Drop columns that have an unexpected value.
    consent_categories = ['good', 'bad']
    if not set(df['consent'].dropna().unique()).issubset(consent_categories):
        print(f"Column 'consent' had unexpected values.")

    print("Check complete.")

check_column_values(data)


def remove_unexpected_consent(df):
    # Define acceptable categories for the 'consent' column
    consent_categories = ['good', 'bad']

    # Create a mask that identifies rows where 'consent' has unexpected or missing values
    unexpected_mask = ~df['consent'].isin(consent_categories) | df['consent'].isna()

    # Remove these rows from the DataFrame
    cleaned_df = df[~unexpected_mask]

    return cleaned_df

# Assuming 'data' is your DataFrame
data = remove_unexpected_consent(data)


#outliers

numeric_columns = data.select_dtypes(include=['number']).drop(columns=['profession'], errors='ignore')

# Plotting the boxplots
plt.figure(figsize=(12, 6))  # Adjust the size to fit all subplots comfortably
numeric_columns.boxplot()
plt.title('Boxplot for Numeric Columns Excluding "Profession"')
plt.xticks(rotation=45)  # Rotate labels to avoid overlap
#plt.show()

non_numeric_columns = ['consent', 'vacation']
non_numeric_data = data[non_numeric_columns]
fig, axs = plt.subplots(1, 2, figsize=(14, 7))

for i, column in enumerate(non_numeric_columns):
    value_counts = non_numeric_data[column].value_counts()
    axs[i].pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=140)
    axs[i].set_title(f'Distribution of {column}')

plt.suptitle('Pie Charts for Non-Numeric Columns: Consent and Vacation')
plt.show()

#Is the column profession unique?
num_unique_professions = data['profession'].nunique()
total_rows = len(data)

# Check if the number of unique values in the 'profession' column is equal to the total number of rows
is_profession_unique = num_unique_professions == total_rows

print(f"Number of unique professions: {num_unique_professions}, Total rows: {total_rows}")

duplicate_mask = data.duplicated(subset=['profession'], keep=False)  # Mark all duplicates

# Display which professions are not unique
non_unique_professions = data[duplicate_mask]['profession'].unique()
print("Non-unique professions:", non_unique_professions)

# Remove all rows that are duplicates in the 'profession' column
# keep='first' to keep the first occurrence, use keep=False to remove all duplicates
data_cleaned = data.drop_duplicates(subset=['profession'], keep='first')

# Check the result
print("Number of rows after removal:", len(data_cleaned))

# Start with Classification - Logical Regressionn


categorical_features = ['vacation']
numeric_features = ['duration', 'wage1', 'wage2', 'hours', 'holidays']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Handling missing values
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Handling missing values
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)])

# Define target and features
X = data.drop('consent', axis=1)
y = data['consent'].map({'good': 1, 'bad': 0})

n_runs = 10

# Store scores
accuracy_scores = []
precision_scores = []
recall_scores = []

accuracy_scores_smot = []
precision_scores_smot = []
recall_scores_smot = []

accuracy_scores_over = []
precision_scores_over = []
recall_scores_over = []

accuracy_scores_under = []
precision_scores_under = []
recall_scores_under = []

print("Without any balancer:")
for i in range(n_runs):
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    # Pipeline without SMOTE
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', LogisticRegression())])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)

    if i == 9:
        print(classification_report(y_test, y_pred))

print("Average Accuracy:", np.mean(accuracy_scores))
print("Average Precision:", np.mean(precision_scores))
print("Average Recall:", np.mean(recall_scores))



# With SMOTE - SMOTE creates synthetic training data for the underrepresented groups

print("With SMOTE:")
for i in range(n_runs):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('smote', SMOTE(random_state=42)),
                            ('classifier', LogisticRegression())])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy_smot = accuracy_score(y_test, y_pred)
    precision_smot = precision_score(y_test, y_pred)
    recall_smot = recall_score(y_test, y_pred)

    accuracy_scores_smot.append(accuracy_smot)
    precision_scores_smot.append(precision_smot)
    recall_scores_smot.append(recall_smot)
    if i == 9:
        print(classification_report(y_test, y_pred))

print("Average Accuracy:", np.mean(accuracy_scores_smot))
print("Average Precision:", np.mean(precision_scores_smot))
print("Average Recall:", np.mean(recall_scores_smot))


# With Oversampling - Oversample the bad consent rows so its more balanced
print("With Oversampling:")
for i in range(n_runs):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('oversampler', RandomOverSampler(random_state=42)),
                            ('classifier', LogisticRegression())])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy_over = accuracy_score(y_test, y_pred)
    precision_over = precision_score(y_test, y_pred)
    recall_over = recall_score(y_test, y_pred)

    accuracy_scores_over.append(accuracy_over)
    precision_scores_over.append(precision_over)
    recall_scores_over.append(recall_over)
    if i == 9:
        print(classification_report(y_test, y_pred))

print("Average Accuracy:", np.mean(accuracy_scores_over))
print("Average Precision:", np.mean(precision_scores_over))
print("Average Recall:", np.mean(recall_scores_over))

#with undersampling the overrepresented data
print("With Undersampling:")
for i in range(n_runs):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('undersampler', RandomUnderSampler(random_state=42)),
                            ('classifier', LogisticRegression())])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy_under = accuracy_score(y_test, y_pred)
    precision_under = precision_score(y_test, y_pred)
    recall_under = recall_score(y_test, y_pred)
    accuracy_scores_under.append(accuracy_under)
    precision_scores_under.append(precision_under)
    recall_scores_under.append(recall_under)
    if i == 9:
        print(classification_report(y_test, y_pred))

print("Average Accuracy:", np.mean(accuracy_scores_under))
print("Average Precision:", np.mean(precision_scores_under))
print("Average Recall:", np.mean(recall_scores_under))
