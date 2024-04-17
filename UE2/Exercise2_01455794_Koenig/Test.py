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
print(frequent_itemsets_multi)

# Filter rules to focus on Product.Name => Sub.Category
# Generate association rules with a more practical confidence threshold
rules_multi = association_rules(frequent_itemsets_multi, metric='confidence', min_threshold=0.01)

# Filter rules to specifically find associations from products to subcategories
filtered_rules1 = rules_multi[
    (rules_multi['antecedents'].apply(lambda x: not any('SubCategory:' in item for item in x))) &
    (rules_multi['consequents'].apply(lambda x: any('SubCategory:' in item for item in x))) &
    (rules_multi['confidence'] < 1)  # Exclude rules where confidence is exactly 1
]

filtered_rules2 = rules_multi[
    (rules_multi['consequents'].apply(lambda x: not any('SubCategory:' in item for item in x))) &
    (rules_multi['antecedents'].apply(lambda x: any('SubCategory:' in item for item in x))) &
    (rules_multi['confidence'] < 1)  # Exclude rules where confidence is exactly 1
]

print('Filtered Multilevel Association Rules:')
print(filtered_rules1[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
print(filtered_rules2[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
