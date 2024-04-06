import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('company_data.csv')
pd.set_option('display.max_columns', None)

#print(data.head(50))

# Task 1: Calculate the age and get the distribution within each marital status
current_year = pd.Timestamp('now').year
data['age'] = current_year - data['year_of_birth']
age_distribution_marital_status = data.groupby('marital_status')['age'].describe()
print(age_distribution_marital_status)

absurd_age_data = data[data['marital_status'] == 'Absurd']['age']
alone_age_data = data[data['marital_status'] == 'Alone']['age']
divorced_age_data = data[data['marital_status'] == 'Divorced']['age']
married_age_data = data[data['marital_status'] == 'Married']['age']
single_age_data = data[data['marital_status'] == 'Single']['age']
together_age_data = data[data['marital_status'] == 'Together']['age']
widow_age_data = data[data['marital_status'] == 'Widow']['age']
yolo_age_data = data[data['marital_status'] == 'YOLO']['age']


plt.figure(figsize=(15, 10))
plt.subplot(2, 4, 1)  # 2 row, 4 columns, 1st subplot
plt.boxplot(absurd_age_data)
plt.title('Boxplot for Martial Status: Absurd')
plt.xlabel('Age')
plt.tight_layout()

plt.subplot(2, 4, 2)
plt.boxplot(alone_age_data)
plt.title('Boxplot for Martial Status: Alone')
plt.xlabel('Age')
plt.tight_layout()

plt.subplot(2, 4, 3)
plt.boxplot(divorced_age_data)
plt.title('Boxplot for Martial Status: Divorced')
plt.xlabel('Age')
plt.tight_layout()

plt.subplot(2, 4, 4)
plt.boxplot(married_age_data)
plt.title('Boxplot for Martial Status: Married')
plt.xlabel('Age')
plt.tight_layout()

plt.subplot(2, 4, 5)
plt.boxplot(single_age_data)
plt.title('Boxplot for Martial Status: Single')
plt.xlabel('Age')
plt.tight_layout()

plt.subplot(2, 4, 6)
plt.boxplot(together_age_data)
plt.title('Boxplot for Martial Status: Together')
plt.xlabel('Age')
plt.tight_layout()

plt.subplot(2, 4, 7)
plt.boxplot(widow_age_data)
plt.title('Boxplot for Martial Status: Widow')
plt.xlabel('Age')
plt.tight_layout()

plt.subplot(2, 4, 8)
plt.boxplot(yolo_age_data)
plt.title('Boxplot for Martial Status: YOLO')
plt.xlabel('Age')
plt.tight_layout()

plt.show()


#Task 2 - What is the distribution of the education?

education_distribution = data['education'].value_counts()
print(education_distribution)

plt.figure(figsize=(8, 8))
education_distribution.plot(kind='pie', autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Education Levels')
plt.ylabel('')
plt.show()

#Task 3 -  Which country has the most web purchases?

country_web_purchases = data.groupby('country')['web_purchases'].sum().idxmax()
web_purchases = data.groupby('country')['web_purchases'].sum().max()
#return the index of the first occurrence of the maximum value over the requested axis
print("\nCountry with the Most Web Purchases:")
print(f'{country_web_purchases} with {web_purchases} purchases')

#for the plot
country_web_purchases_sorted = data.groupby('country')['web_purchases'].sum().sort_values(ascending=False)
top_countries = country_web_purchases_sorted.head(5)
plt.figure(figsize=(10, 6))
top_countries.plot(kind='bar', color='skyblue')
plt.title('Top 5 Countries by Web Purchases')
plt.xlabel('Country')
plt.ylabel('Total Web Purchases')
plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
plt.show()

#Task 4 -How does the average customer look like?

#Make the average for all columns except customerNr and year of birth (drop these two columns (axis 0 would be rows)
numeric_columns = data.select_dtypes(include=['int', 'float'])
average_customer_numerical = numeric_columns.drop(['customerNr', 'year_of_birth'], axis=1).mean()

#Takes the median of the attributes education, martial status und country - and takes always the firt one - iloc[0]
average_customer_categorical = data[['education', 'marital_status', 'country']].mode().iloc[0]

average_customer_profile = pd.concat([average_customer_numerical, average_customer_categorical])
print(average_customer_profile)

#for plot
#only those values will be shown that are bigger than 1 to make it more readable
average_numerical  = {key: val for key, val in average_customer_numerical.items() if val > 1}
attributes = list(average_numerical.keys())
average_values = list(average_numerical.values())

plt.figure(figsize=(10, 6))
plt.bar(attributes, average_values, color='skyblue')
plt.xlabel('Attributes')
plt.ylabel('Average Value')
plt.title('Average Customer Numerical Profile')
plt.xticks(rotation=45, ha="right")  # Rotate the attribute names for better readability
plt.tight_layout()
plt.show()

#Task 5 - Which previous marketing campaign was most successful?

campaign_columns = ['campaign1', 'campaign2', 'campaign3', 'campaign4', 'campaign5']

campaign_success = data[campaign_columns].sum()
most_successful_campaign = campaign_success.idxmax()

most_successful_campaign_responses = campaign_success.max()
#max returns the highest value, idmax the attribute not the value itself

print("Most Successful Campaign:", most_successful_campaign)
print("Number of Responses:", most_successful_campaign_responses)


#for the plot
success_sorted = campaign_success.sort_values(ascending=False)
print(success_sorted)
plt.figure(figsize=(10, 10))
success_sorted.plot(kind='bar', color='skyblue')
plt.title('Top 3 Most Successfully Campaigns')
plt.xlabel('Campaigns')
plt.ylabel('Total Number of Responses')
plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
plt.show()

#overall description of the data

#Geographical Analysis for purchases
data['overall_purchases'] = data['web_purchases']+data['store_purchases']

overall_purchases_per_country = data.groupby('country')['overall_purchases'].sum()
overall_purchases_per_country_sorted = overall_purchases_per_country.sort_values(ascending=False)

store_purchases_per_country = data.groupby('country')['store_purchases'].sum()
store_purchases_per_country_sorted = store_purchases_per_country.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
overall_purchases_per_country_sorted.plot(kind='bar', color='green')
plt.title('Total Overall Purchases per Country')
plt.xlabel('Country')
plt.ylabel('Total Web Purchases')
plt.xticks(rotation=45)

plt.figure(figsize=(10, 6))
store_purchases_per_country_sorted.plot(kind='bar', color='green')
plt.title('Total Store Purchases per Country')
plt.xlabel('Country')
plt.ylabel('Total Store Purchases')
plt.xticks(rotation=45)
plt.show()

#Income Analysis for web purchases and normal purchases
data['income'] = data['income'].str.replace(',', '').str.replace('$', '')

data['income'] = pd.to_numeric(data['income'], errors='coerce')
data['web_purchases'] = pd.to_numeric(data['web_purchases'], errors='coerce')
data['store_purchases'] = pd.to_numeric(data['store_purchases'], errors='coerce')

#webpurchases
plt.figure(figsize=(10,6))
plt.scatter(data['income'], data['web_purchases'], color='blue', alpha=0.5, marker='x')
plt.title("Income & Number of Web Purchases")
plt.xlabel("Income")
plt.ylabel("Web Purchases")
plt.grid(True)

#store purchases
plt.figure(figsize=(10,6))
plt.scatter(data['income'], data['store_purchases'], color='red', alpha=0.5, marker='o')
plt.title("Income & Number of Store Purchases")
plt.xlabel("Income")
plt.ylabel("Store Purchases")
plt.grid(True)

#both combined
plt.figure(figsize=(10,6))
plt.scatter(data['income'], data['web_purchases'], color='blue', alpha=0.5, marker='x', label='Web Purchases')
plt.scatter(data['income'], data['store_purchases'], color='red', alpha=0.5, marker='o', label='Store Purchases')
plt.title("Income in combination with the Number of Web Purchases and Number of Store Purchases")
plt.xlabel("Income")
plt.ylabel("Web & Store Purchases")
plt.grid(True)
plt.legend()
plt.show()

#Age Analysis
mean_age = data['age'].mean()

older_customers = data[data['age'] > mean_age]
younger_customers = data[data['age'] <= mean_age]

plt.figure(figsize=(10, 6))

# Boxplot for customers older than the mean age
plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
plt.boxplot(older_customers['last_purchase_in_days'])
plt.title(f'Older than Mean Age ({mean_age:.2f} years)')
plt.ylabel('Days Since Last Purchase')

# Boxplot for customers younger than or equal to the mean age
plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
plt.boxplot(younger_customers['last_purchase_in_days'])
plt.title(f'Younger than Mean Age ({mean_age:.2f} years)')
plt.ylabel('Days Since Last Purchase')
plt.tight_layout()


plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
plt.boxplot(older_customers['overall_purchases'])
plt.title(f'Older than Mean Age ({mean_age:.2f} years)')
plt.ylabel('Overall Purchases')

plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
plt.boxplot(older_customers['overall_purchases'])
plt.title(f'Younger than Mean Age ({mean_age:.2f} years)')
plt.ylabel('Overall Purchases')
plt.tight_layout()


plt.figure(figsize=(10,6))
plt.scatter(data['overall_purchases'], data['age'], color='blue', alpha=0.5, marker='x')
plt.title("Overall Purchases and Age")
plt.xlabel("Overall Purchases")
plt.ylabel("Age")
plt.grid(True)
plt.legend()
plt.show()

# #Data Quality is questionable due to people being over 120 years old