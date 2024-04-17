import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

#reading couldn't be done with the standard  UTF-8 encoding
data = pd.read_csv("store.csv", encoding='ISO-8859-1')

pd.set_option('display.max_columns', None)

#print(data['Postal.Code'].head(10))

#clean data

missing_values_count = data.isnull().sum()
missing_percentage = (missing_values_count / len(data)) * 100
print(missing_percentage)
#Postal Code is in over 80% missing, the column will therefore be dropped

data = data.drop('Postal.Code', axis=1)

placeholder_values = [666, 999]
mask = data.isin(placeholder_values)
#print(mask.sum())
#One 666 in Sales

x = data['Sales'].isin(placeholder_values)

filtered_data = data[x]

if not filtered_data.empty:
    print("Found rows with the placeholder values in the 'Sales' column:")
    print(filtered_data[['Row.ID', 'Sales']])
else:
    print("No rows found with the placeholder values in the 'Sales' column.")

#sales = data['Sales'].describe()

median_sales = data.loc[~data['Sales'].isin(placeholder_values), 'Sales'].median()
data['Sales'].replace(placeholder_values, median_sales, inplace=True)

missing_values_count = data.isnull().sum()
print(f"Number of still missing values: {missing_values_count}")

#look for outliers

numeric_data = data.select_dtypes(include=['number']).drop('Row.ID', axis=1)
numeric_data.plot(kind='box', subplots=True, layout=(5,2), figsize=(20, 20))
plt.tight_layout()  # Adjusts plot to ensure it fits into the figure area nicely
#plt.show()
#negative Profit? - in my interpretation it's possible and signalisies a loss

#explore the data

subcategory_counts = data['Sub.Category'].value_counts()
top_subcategories = subcategory_counts.head(10)

plt.figure(figsize=(10, 6))
top_subcategories.plot(kind='bar', color='skyblue')
plt.title('Top 10 Subcategories by Occurrence')
plt.xlabel('Subcategory')
plt.ylabel('Number of Occurrences')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

product_counts = data['Product.Name'].value_counts()
top_products = product_counts.head(10)

plt.figure(figsize=(10, 6))
top_products.plot(kind='bar', color='skyblue')
plt.title('Top 10 Products by Occurrence')
plt.xlabel('Product')
plt.ylabel('Number of Occurrences')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

subcategory_distribution_customer = data.groupby('Customer.Name')['Sub.Category'].describe()
sorted_symptoms = subcategory_distribution_customer.sort_values(by='freq', ascending=False)
print(sorted_symptoms.head(10))

#Do the Pattern Mining Algorithm (A-Priori)

#print(data.columns)

# Subcategories per Order_ID

data_pivot_1 = data.pivot_table(index='Order.ID', columns='Sub.Category', aggfunc='size', fill_value=0)

# Convert to 1s and 0s, where 1 indicates presence of the product in the transaction
data_encoded_1 = (data_pivot_1 > 0).astype(int)
print(data_encoded_1.head(10))

frequent_itemsets_1 = apriori(data_encoded_1, min_support=0.023, use_colnames=True)

# Generate association rules
rules_1 = association_rules(frequent_itemsets_1, metric='confidence', min_threshold=0.15)


#Subcategories per Customer

data_pivot_2 = data.pivot_table(index='Customer.ID', columns='Sub.Category', aggfunc='size', fill_value=0)

# Convert to 1s and 0s, where 1 indicates presence of the product in the transaction
data_encoded_2 = (data_pivot_2 > 0).astype(int)

frequent_itemsets_2 = apriori(data_encoded_2, min_support=0.7, use_colnames=True)

# Generate association rules
rules_2 = association_rules(frequent_itemsets_2, metric='confidence', min_threshold=0.9)



#Lower confidence and min_support lead to extremly large lisst - especially a low min_support couldn't even be computed properly

#Products bought on same day

data_pivot_3 = data.pivot_table(index='Order.Date', columns='Sub.Category', aggfunc='size', fill_value=0)

# Convert to 1s and 0s, where 1 indicates presence of the product in the transaction
data_encoded_3 = (data_pivot_3 > 0).astype(int)


frequent_itemsets_3 = apriori(data_encoded_3, min_support=0.74, use_colnames=True)

# Generate association rules
rules_3 = association_rules(frequent_itemsets_3, metric='confidence', min_threshold=0.9)

# help of chatgpt
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

# Enhance rules with new metrics
enhanced_rules_1 = add_kulczynski_and_imbalance(rules_1, frequent_itemsets_1)
print('A-Priori Rules for buying Products of a Subcategory together in one order:')
print(enhanced_rules_1[['antecedents', 'consequents', 'support', 'confidence', 'lift', 'kulczynski', 'imbalance_ratio']])

enhanced_rules_2 = add_kulczynski_and_imbalance(rules_2, frequent_itemsets_2)
print('A-Priori Rules for buying Products of a Subcategory together from one customer:')
print(enhanced_rules_2[['antecedents', 'consequents', 'support', 'confidence', 'lift', 'kulczynski', 'imbalance_ratio']])

enhanced_rules_3 = add_kulczynski_and_imbalance(rules_3, frequent_itemsets_3)
print('A-Priori Rules for buying Products of a Subcategory together on the same day:')
print(enhanced_rules_3[['antecedents', 'consequents', 'support', 'confidence', 'lift', 'kulczynski', 'imbalance_ratio']])

#Multilevel Assosiations from SubCategory to Category

# Creating item identifiers for both categories (converted to string) and subcategories
data['Category_Item'] = "Category:" + data['Category'].astype(str)  # Convert categories to string and prefix
data['SubCategory_Item'] = "SubCategory:" + data['Sub.Category']  # Assuming Sub.Category is already a string

# Collect lists per order, including both categories and subcategory tags
grouped = data.groupby('Order.ID').apply(lambda x: x['Category_Item'].tolist() + x['SubCategory_Item'].tolist())

# Use TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit_transform(grouped)
df = pd.DataFrame(te_ary, columns=te.columns_)
print(df)

# Generate frequent itemsets
frequent_itemsets = apriori(df, min_support=0.06, use_colnames=True)
# print(frequent_itemsets)

# Generate association rules with a practical confidence threshold
rules_multi_1 = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.5)

# Filter rules to specifically find associations from subcategories to categories
filtered_rules_1 = rules_multi_1[
    (rules_multi_1['consequents'].apply(lambda x: all('Category:' in item for item in x))) &
    (rules_multi_1['antecedents'].apply(lambda x: any('SubCategory:' in item for item in x))) &
    (rules_multi_1['confidence'] < 1)  # Exclude rules where confidence is exactly 1
]

print('Filtered Association Rules from Subcategories to Categories:')
print(filtered_rules_1[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# Multilevel Assosiations from Product.Name to Sub.Category

data['Item'] = data['Product.Name']
data['SubCategory_Item'] = "SubCategory:" + data['Sub.Category']

# Collect lists per order, ensuring both products and subcategory tags are included
grouped = data.groupby('Order.ID').apply(lambda x: x['Item'].tolist() + x['SubCategory_Item'].tolist())

# Use TransactionEncoder

te = TransactionEncoder()
te_ary = te.fit_transform(grouped)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Generate association rules
frequent_itemsets_multi = apriori(df, min_support=0.0015, use_colnames=True)  # Adjust as needed based on data
# print(frequent_itemsets_multi)

# Filter rules to focus on Product.Name => Sub.Category
# Generate association rules with a more practical confidence threshold
rules_multi_2 = association_rules(frequent_itemsets_multi, metric='confidence', min_threshold=0.1)

# Filter rules to specifically find associations from products to subcategories
filtered_rules_2 = rules_multi_2[
    (rules_multi_2['antecedents'].apply(lambda x: not any('SubCategory:' in item for item in x))) &
    (rules_multi_2['consequents'].apply(lambda x: any('SubCategory:' in item for item in x))) &
    (rules_multi_2['confidence'] < 1)  # Exclude rules where confidence is exactly 1
]

print('Filtered Multilevel Association Rules Product --> Subcategory:')
print(filtered_rules_2[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

#redundancy check

#help of ChatGPT
def remove_redundant_rules(rules):
    # Sort rules by confidence and support for prioritization
    rules = rules.sort_values(by=['confidence', 'support'], ascending=False)

    # Dictionary to hold the best rule for each antecedent set
    best_rules = {}

    for _, rule in rules.iterrows():
        antecedents = frozenset(rule['antecedents'])
        consequents = frozenset(rule['consequents'])

        if antecedents in best_rules:
            # Check if current rule's consequents are a subset of any existing rule with the same antecedents
            if consequents <= best_rules[antecedents]['consequents']:
                continue  # Skip this rule as it is redundant
            elif consequents > best_rules[antecedents]['consequents']:
                # If current rule has more consequents, update the best rule
                best_rules[antecedents] = {'consequents': consequents, 'record': rule}
        else:
            best_rules[antecedents] = {'consequents': consequents, 'record': rule}

    # Extracting the non-redundant rules
    non_redundant_rules = pd.DataFrame([val['record'] for val in best_rules.values()])
    return non_redundant_rules

non_redundant_rules_1 = remove_redundant_rules(filtered_rules_1)
print('Non-Redundant Association Rules Subcategory --> Category:')
print(non_redundant_rules_1[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

non_redundant_rules_2 = remove_redundant_rules(filtered_rules_2)
print('Non-Redundant Association Rules Product --> Subcategory:')
print(non_redundant_rules_2[['antecedents', 'consequents', 'support', 'confidence', 'lift']])