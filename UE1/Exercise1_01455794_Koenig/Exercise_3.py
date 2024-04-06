import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway

data = pd.read_csv('cities.csv')
pd.set_option('display.max_columns', None)

data_df = pd.DataFrame(data)
#print(data.head(50))

#replace missing numeric values with the median
data_df['Quality.of.Life'] = data_df['Quality.of.Life'].fillna(data_df['Quality.of.Life'].median())
data_df['Crime.Rating'] = data_df['Crime.Rating'].fillna(data_df['Crime.Rating'].median())
data_df['Pollution'] = data_df['Pollution'].fillna(data_df['Pollution'].median())
data_df['Purchase.Power'] = data_df['Purchase.Power'].fillna(data_df['Purchase.Power'].median())

# kde=True --> plot a KDE curve on top of the histogram

sns.histplot(data_df['Quality.of.Life'], kde=True, color="blue", label="Quality of Life")
sns.histplot(data_df['Crime.Rating'], kde=True, color="red", label="Crime Rating")
plt.title('Life Quality x Crime Rating')
plt.ylabel('City Count')
plt.xlabel('Value (%)')
plt.legend()
#plt.show()

plt.figure(figsize=(10,6))
plt.scatter(data_df['Quality.of.Life'], data['Crime.Rating'], color='blue', alpha=0.5, marker='x')
plt.title("Life Quality x Crime Rating")
plt.xlabel("Life Quality")
plt.ylabel("Crime Rating")
plt.grid(True)
#plt.show()
# In the Scatterplot you don't really see a correlation between these two attributes - in the histogram however you do

corr_coefficient_crime= data['Crime.Rating'].corr(data['Quality.of.Life'])
covariance_crime = data['Crime.Rating'].cov(data['Quality.of.Life'])

#slight - moderate negative correlation between crime and life quality and negative covariance

print("Pearson Correlation Coefficient Crime Rating x Quality of Life:", corr_coefficient_crime)
print("Covariance Crime Rating x Quality of Life:", covariance_crime)

# Binning Quality of Life scores into 10 - 3 bins

bin_numbers = list(range(3, 11))

plt.figure(figsize=(20, 20))

# Iterating over the specified bin numbers to create subplots
for i, bins in enumerate(bin_numbers, start=1):
    plt.subplot(4, 2, i)
    sns.histplot(data_df['Quality.of.Life'], bins=bins, kde=True, color='skyblue')
    labels = range(1, bins + 1)
    data_df['QualityGroup'] = pd.cut(data_df['Quality.of.Life'], bins=bins, labels=labels)
    plt.title(f'{bins} Bins')

    formatted_output = f"Range of Life Quality in {bins} Bins:\n\n"
    x = bins + 1
    min_quality_by_bin = data_df.groupby('QualityGroup')['Quality.of.Life'].min()
    max_quality_by_bin = data_df.groupby('QualityGroup')['Quality.of.Life'].max()
    mean_quality_by_bin = data_df.groupby('QualityGroup')['Quality.of.Life'].mean()

    for bin in range(1, x):
        min_value = min_quality_by_bin.loc[bin]
        max_value = max_quality_by_bin.loc[bin]
        mean_value = mean_quality_by_bin.loc[bin]
        bin_data = data_df[data_df['QualityGroup'] == bin]
        correlation = bin_data['Quality.of.Life'].corr(bin_data['Crime.Rating'])
        formatted_output += f"Bin {bin}: {min_value:.2f} to {max_value:.2f}, Mean: {mean_value:.2f}, Correlation: {correlation: .5f}\n"

    print(formatted_output)

plt.tight_layout()
plt.show()


#for simplicity reasons we will calculate it with 3, 5 and 10 bins
data_df['QualityGroup_5'] = pd.cut(data_df['Quality.of.Life'], bins=5, labels=range(1, 6))
data_df['QualityGroup_10'] = pd.cut(data_df['Quality.of.Life'], bins=10, labels=range(1, 11))
data_df['QualityGroup_3'] = pd.cut(data_df['Quality.of.Life'], bins=3, labels=range(1, 4))
crime_rating_by_quality_bin_5 = data_df.groupby('QualityGroup_5')['Crime.Rating'].mean()
crime_rating_by_quality_bin_10 = data_df.groupby('QualityGroup_10')['Crime.Rating'].mean()
crime_rating_by_quality_bin_3 = data_df.groupby('QualityGroup_3')['Crime.Rating'].mean()


plt.figure(figsize=(10, 6))
crime_rating_by_quality_bin_5.plot(kind='bar', color='orange')
plt.title('Average Crime Rating by Quality of Life with 5 Bins')
plt.xlabel('Quality of Life Bin (Low - High)')
plt.ylabel('Average Crime Rating')
plt.xticks(rotation=0)

plt.figure(figsize=(10, 6))
crime_rating_by_quality_bin_10.plot(kind='bar', color='orange')
plt.title('Average Crime Rating by Quality of Life with 10 Bins')
plt.xlabel('Quality of Life Bin (Low - High)')
plt.ylabel('Average Crime Rating')
plt.xticks(rotation=0)

plt.figure(figsize=(10, 6))
crime_rating_by_quality_bin_3.plot(kind='bar', color='orange')
plt.title('Average Crime Rating by Quality of Life with 3 Bins')
plt.xlabel('Quality of Life Bin (Low - High)')
plt.ylabel('Average Crime Rating')
plt.xticks(rotation=0)
plt.show()

# Correlation Analysis within each Quality of Life bin
print('5 Bins:')
for bin_label in data_df['QualityGroup_5'].unique():
    bin_data = data_df[data_df['QualityGroup_5'] == bin_label]
    correlation = bin_data['Quality.of.Life'].corr(bin_data['Crime.Rating'])
    print(f"Correlation in Bin {bin_label}: {correlation}")

print('10 Bins:')
for bin_label in data_df['QualityGroup_10'].unique():
    bin_data = data_df[data_df['QualityGroup_10'] == bin_label]
    correlation = bin_data['Quality.of.Life'].corr(bin_data['Crime.Rating'])
    print(f"Correlation in Bin {bin_label}: {correlation}")

print('3 Bins:')
for bin_label in data_df['QualityGroup_3'].unique():
    bin_data = data_df[data_df['QualityGroup_3'] == bin_label]
    correlation = bin_data['Quality.of.Life'].corr(bin_data['Crime.Rating'])
    print(f"Correlation in Bin {bin_label}: {correlation}")

#other analysis
plt.figure(figsize=(10,6))
plt.scatter(data['Pollution'], data['Quality.of.Life'], color='blue', alpha=0.5, marker='x')
plt.title("Pollution & Life Quality")
plt.xlabel("Pollution")
plt.ylabel("Life Quality (%)")
plt.grid(True)
plt.show()
#no real result

plt.figure(figsize=(10,6))
plt.scatter(data['Purchase.Power'], data['Quality.of.Life'], color='blue', alpha=0.5, marker='x')
plt.title("Purchase.Power & Life Quality")
plt.xlabel("Purchase.Power")
plt.ylabel("Life Quality (%)")
plt.grid(True)
plt.show()
#here a trend can be seen

corr_coefficient_pur = data['Purchase.Power'].corr(data['Quality.of.Life'])
covariance_pur = data['Purchase.Power'].cov(data['Quality.of.Life'])

corr_coefficient_pol = data['Pollution'].corr(data['Quality.of.Life'])
covariance_pol = data['Pollution'].cov(data['Quality.of.Life'])

print("Pearson Correlation Coefficient Purchase Power x Quality of Life:", corr_coefficient_pur)
print("Covariance Purchase Power x Quality of Life:", covariance_pur)

#high positive correlation and positive covariance between purchase power and life quality

print("Pearson Correlation Coefficient Pollution x Quality of Life:", corr_coefficient_pol)
print("Covariance Pollution x Quality of Life:", covariance_pol)

#slight negative correlation and negative covariance between pollution and life quality