import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# reading couldn't be done with the standard  UTF-8 encoding
bad_lines_count = 0

#filter out the lines with more arguments than columns
def handle_bad_lines(line):
    global bad_lines_count
    bad_lines_count += 1
    #print("Bad line:", line)

try:
    data = pd.read_csv("health.csv", encoding='ISO-8859-1', on_bad_lines=handle_bad_lines, engine='python')
    print("Data loaded successfully. Here's a preview:")
    print(data.head())
except Exception as e:
    print("Failed to load data:", e)

data.rename(columns={'Unnamed: 0': 'ID'}, inplace=True)

pd.set_option('display.max_columns', None)

print(f'Skipped Lines: {bad_lines_count}')
row_count = len(data)
print(f"Total number successful inserted lines: {row_count}")

#clean data

missing_values_count = data.isnull().sum()
print(f"Number of missing values before cleaning:\n{missing_values_count}")

for column in data.columns:
    if pd.api.types.is_numeric_dtype(data[column]):
        median = data[column].median()
        data.fillna({column: median}, inplace=True)
    else:
        most_common = data[column].mode()[0]
        data.fillna({column: most_common}, inplace=True)

missing_values_count = data.isnull().sum()
print(f"Number of missing values after cleaning:\n{missing_values_count}")

placeholder_values = [666, 999]
mask = data.isin(placeholder_values)
print(mask.sum())
mask_id = data['ID'].isin(placeholder_values)
mask_reportid = data['reportid'].isin(placeholder_values)

# Use masks to filter data
filtered_data_id = data[mask_id]
filtered_data_reportid = data[mask_reportid]

# Check if filtered dataframes are empty and print results
if not filtered_data_id.empty:
    print("Found rows with the placeholder values in the 'ID' column:")
    print(filtered_data_id[['ID']])
else:
    print("No rows found with the placeholder values in the 'ID' column.")

if not filtered_data_reportid.empty:
    print("Found rows with the placeholder values in the 'reportid' column:")
    print(filtered_data_reportid[['reportid']])
else:
    print("No rows found with the placeholder values in the 'reportid' column.")

data = data[~mask_id]
data = data[~mask_reportid]

row_count = len(data)
print(f"Total number cleaned dataset: {row_count}")

#explore data
symptom_data = data[data['type'] == 'S']
weather_data = data[data['type'] == 'W']
note_data = data[data['type'] == 'N']
food_data = data[data['type'] == 'F']
condition_data = data[data['type'] == 'C']
treatment_data = data[data['type'] == 'T']

symptom_counts = symptom_data['name'].value_counts().head(5)
weather_counts = weather_data['name'].value_counts().head(5)
note_counts = note_data['name'].value_counts().head(5)
food_counts = food_data['name'].value_counts().head(5)
condition_counts = condition_data['name'].value_counts().head(5)
treatment_counts = treatment_data['name'].value_counts().head(5)

fig, axs = plt.subplots(3, 2, figsize=(14, 10))
axs[0, 0].bar(symptom_counts.index, symptom_counts.values, color='skyblue')
axs[0, 0].set_title('Top 5 Symptoms')
axs[0, 0].set_xticklabels(symptom_counts.index, rotation=45)

axs[0, 1].bar(weather_counts.index, weather_counts.values, color='orange')
axs[0, 1].set_title('Top 5 Weather Types')
axs[0, 1].set_xticklabels(weather_counts.index, rotation=45)

axs[1, 0].bar(note_counts.index, note_counts.values, color='green')
axs[1, 0].set_title('Top 5 Notes')
axs[1, 0].set_xticklabels(note_counts.index, rotation=45)

axs[1, 1].bar(food_counts.index, food_counts.values, color='red')
axs[1, 1].set_title('Top 5 Foods')
axs[1, 1].set_xticklabels(food_counts.index, rotation=45)

axs[2, 0].bar(condition_counts.index, condition_counts.values, color='purple')
axs[2, 0].set_title('Top 5 Conditions')
axs[2, 0].set_xticklabels(condition_counts.index, rotation=45)

axs[2, 1].bar(treatment_counts.index, treatment_counts.values, color='cyan')
axs[2, 1].set_title('Top 5 Treatments')
axs[2, 1].set_xticklabels(treatment_counts.index, rotation=45)

plt.tight_layout()
plt.show()

#average transactions and max user

#symtomps
numeric_columns = symptom_data.select_dtypes(include=['int', 'float'])
average_transaction_numerical = numeric_columns.drop(['ID', 'reportid'], axis=1).mean()

average_transaction_categorical = symptom_data[['user', 'date', 'name', 'value']].mode().iloc[0]

average_transaction_profile = pd.concat([average_transaction_numerical, average_transaction_categorical])
print(f"Average symptom transaction: \n{average_transaction_profile}")

symptom_distribution_user = symptom_data.groupby('user')['name'].describe()
sorted_symptoms = symptom_distribution_user.sort_values(by='freq', ascending=False)
print(f"Top 5 users and their reported symptoms: \n{sorted_symptoms.head(5)}")

#weather
numeric_columns = weather_data.select_dtypes(include=['int', 'float'])
average_transaction_numerical = numeric_columns.drop(['ID', 'reportid'], axis=1).mean()

average_transaction_categorical = weather_data[['user', 'date', 'name', 'value']].mode().iloc[0]

average_transaction_profile = pd.concat([average_transaction_numerical, average_transaction_categorical])
print(f"Average weather transaction: \n{average_transaction_profile}")

weather_distribution_user = weather_data.groupby('user')['name'].describe()
sorted_weather = weather_distribution_user.sort_values(by='freq', ascending=False)
print(f"Top 5 users and their reported weather: \n{sorted_weather.head(5)}")

#food
numeric_columns = food_data.select_dtypes(include=['int', 'float'])
average_transaction_numerical = numeric_columns.drop(['ID', 'reportid'], axis=1).mean()

average_transaction_categorical = food_data[['user', 'date', 'name', 'value']].mode().iloc[0]

average_transaction_profile = pd.concat([average_transaction_numerical, average_transaction_categorical])
print(f"Average food transaction: \n{average_transaction_profile}")

food_distribution_user = food_data.groupby('user')['name'].describe()
sorted_food = food_distribution_user.sort_values(by='freq', ascending=False)
print(f"Top 5 users and their reported foods: \n{sorted_food.head(5)}")

#condition
numeric_columns = condition_data.select_dtypes(include=['int', 'float'])
average_transaction_numerical = numeric_columns.drop(['ID', 'reportid'], axis=1).mean()

average_transaction_categorical = condition_data[['user', 'date', 'name', 'value']].mode().iloc[0]

average_transaction_profile = pd.concat([average_transaction_numerical, average_transaction_categorical])
print(f"Average condition transaction: \n{average_transaction_profile}")

condition_distribution_user = condition_data.groupby('user')['name'].describe()
sorted_condition = condition_distribution_user.sort_values(by='freq', ascending=False)
print(f"Top 5 users and their reported conditions: \n{sorted_condition.head(5)}")

#notes
numeric_columns = note_data.select_dtypes(include=['int', 'float'])
average_transaction_numerical = numeric_columns.drop(['ID', 'reportid'], axis=1).mean()

average_transaction_categorical = note_data[['user', 'date', 'name', 'value']].mode().iloc[0]

average_transaction_profile = pd.concat([average_transaction_numerical, average_transaction_categorical])
print(f"Average note transaction: \n{average_transaction_profile}")

note_distribution_user = note_data.groupby('user')['name'].describe()
sorted_note = note_distribution_user.sort_values(by='freq', ascending=False)
print(f"Top 5 users and their notes: \n{sorted_note.head(5)}")

#treatment
numeric_columns = treatment_data.select_dtypes(include=['int', 'float'])
average_transaction_numerical = numeric_columns.drop(['ID', 'reportid'], axis=1).mean()

average_transaction_categorical = treatment_data[['user', 'date', 'name', 'value']].mode().iloc[0]

average_transaction_profile = pd.concat([average_transaction_numerical, average_transaction_categorical])
print(f"Average treatment transaction: \n{average_transaction_profile}")

treatment_distribution_user = treatment_data.groupby('user')['name'].describe()
sorted_treatment = treatment_distribution_user.sort_values(by='freq', ascending=False)
print(f"Top 5 users and their reported treatments: \n{sorted_treatment.head(5)}")

# Do the Pattern Mining Algorithm (A-Priori)

#ChatGPT helped here
def add_kulczynski_and_imbalance(rules, frequent_itemsets):
    # Create a dictionary for quick support value lookup
    support_dict = frequent_itemsets.set_index('itemsets')['support'].to_dict()

    # Calculate Kulczynski and Imbalance Ratio for each rule
    kulczynski = []
    imbalance_ratio = []
    for _, row in rules.iterrows():
        support_A = support_dict[frozenset(row['antecedents'])]
        support_B = support_dict[frozenset(row['consequents'])]
        support_AB = row['support']

        # Kulczynski Calculation
        kulcz = 0.5 * (support_AB / support_A + support_AB / support_B)
        kulczynski.append(kulcz)

        # Imbalance Ratio Calculation
        imbalance = abs(support_A - support_B) / (support_A + support_B - support_AB)
        imbalance_ratio.append(imbalance)

    # Append these lists as new columns to the rules DataFrame
    rules['kulczynski'] = kulczynski
    rules['imbalance_ratio'] = imbalance_ratio

    return rules

# Symptoms that occur together

data_pivot_symptoms = symptom_data.pivot_table(index='reportid', columns='name', aggfunc='size', fill_value=0)

# Convert to 1s and 0s, where 1 indicates presence of the product in the transaction
data_encoded_symptoms = (data_pivot_symptoms > 0).astype(int)

frequent_itemsets_symptoms = apriori(data_encoded_symptoms, min_support=0.1, use_colnames=True)
#print(frequent_itemsets_symptoms)

# Generate association rules
rules_symptoms = association_rules(frequent_itemsets_symptoms, metric='confidence', min_threshold=0.5)


enhanced_rules_symptoms = add_kulczynski_and_imbalance(rules_symptoms, frequent_itemsets_symptoms)
print('A-Priori Rules for symptoms that occur together:')
print(enhanced_rules_symptoms[['antecedents', 'consequents', 'support', 'confidence', 'lift', 'kulczynski', 'imbalance_ratio']])

# Conditions that occur together

data_pivot_condition = condition_data.pivot_table(index='reportid', columns='name', aggfunc='size', fill_value=0)

# Convert to 1s and 0s, where 1 indicates presence of the product in the transaction
data_encoded_condition = (data_pivot_condition > 0).astype(int)

frequent_itemsets_condition = apriori(data_encoded_condition, min_support=0.05, use_colnames=True)
#print(frequent_itemsets_condition)

# Generate association rules
rules_condition = association_rules(frequent_itemsets_condition, metric='confidence', min_threshold=0.4)

enhanced_rules_conditions = add_kulczynski_and_imbalance(rules_condition, frequent_itemsets_condition)
print('A-Priori Rules for condition that occur together:')
print(enhanced_rules_conditions[['antecedents', 'consequents', 'support', 'confidence', 'lift', 'kulczynski', 'imbalance_ratio']])

# Treatment that occur together

data_pivot_treatment = treatment_data.pivot_table(index='reportid', columns='name', aggfunc='size', fill_value=0)

# Convert to 1s and 0s, where 1 indicates presence of the product in the transaction
data_encoded_treatment = (data_pivot_treatment > 0).astype(int)

frequent_itemsets_treatment = apriori(data_encoded_treatment, min_support=0.03, use_colnames=True)
#print(frequent_itemsets_treatment)

# Generate association rules
rules_treatment = association_rules(frequent_itemsets_treatment, metric='confidence', min_threshold=0.05)

enhanced_rules_treatment = add_kulczynski_and_imbalance(rules_treatment, frequent_itemsets_treatment)
print('A-Priori Rules for treatments that occur together:')
print(enhanced_rules_treatment[['antecedents', 'consequents', 'support', 'confidence', 'lift', 'kulczynski', 'imbalance_ratio']])

# Multilevel Assosiations

#ChatGPT gave an idea how to solve this
def apply_apriori_for_two_types(data, type1, type2, min_support, min_confidence):
    # Filter data for the specific types
    filtered_data = data[data['type'].isin([type1, type2])].copy()
    filtered_data['item'] = filtered_data['type'] + ':' + filtered_data['name']
    transactions = filtered_data.groupby('reportid')['item'].agg(set).tolist()

    # Transaction encoding
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    # Apriori algorithm
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    # print(frequent_itemsets)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

    # Filter rules to ensure antecedents are from type1 and consequents from type2
    rules_filtered = rules[
        rules['antecedents'].apply(lambda ants: all(ant.split(':')[0] == type1 for ant in ants)) &
        rules['consequents'].apply(lambda cons: all(con.split(':')[0] == type2 for con in cons))
    ]

    return rules_filtered

condition_treatment_rules = apply_apriori_for_two_types(data, 'C', 'T', 0.014, 0.5)
print("Condition to Treatment Rules:")
print(condition_treatment_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

treatment_condition_rules = apply_apriori_for_two_types(data, 'T', 'C', 0.014, 0.9)
print("Treatment To Condition Rules:")
print(treatment_condition_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

weather_condition_rules = apply_apriori_for_two_types(data, 'W', 'C', 0.06, 0.2)
print("Weather To Condition Rules:")
print(weather_condition_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

#Condition to food rules couldn't be computed due to time and/or computer limits (min_support had to be too low and the
#itemset therefore too big to be computed properly).

food_condition_rules = apply_apriori_for_two_types(data, 'F', 'C', 0.005, 0.2)
print("Food To Condition Rules:")
print(food_condition_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])


