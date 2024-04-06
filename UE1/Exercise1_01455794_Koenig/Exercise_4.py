import pandas as pd
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns

voest_df = pd.DataFrame(pd.read_csv('TL0.DE.csv'))
tesla_df = pd.DataFrame(pd.read_csv('VOE.VI.csv'))

pd.set_option('display.max_columns', None)

#print(voest_df.head())
#print(tesla_df.head())

#Convert the Date into Days

voest_df['Date'] = pd.to_datetime(voest_df['Date'])
tesla_df['Date'] = pd.to_datetime(tesla_df['Date'])

earliest_date_voest = pd.to_datetime(voest_df['Date'].min())

earliest_date_tesla = pd.to_datetime(tesla_df['Date'].min())

voest_df['Days'] = (voest_df['Date'] - earliest_date_voest).dt.days + 1
tesla_df['Days'] = (tesla_df['Date'] - earliest_date_tesla).dt.days + 1



#Normalize the data
voest_df['Z-Normalized'] = zscore(voest_df['Close'], axis=0, ddof=1)
tesla_df['Z-Normalized'] = zscore(tesla_df['Close'], axis=0, ddof=1)

voest_df['Index-Normalized'] = voest_df['Close'] / voest_df['Close'].iloc[0]
tesla_df['Index-Normalized'] = tesla_df['Close'] / tesla_df['Close'].iloc[0]

print('Voest:')
print(voest_df[['Days', 'Close', 'Z-Normalized', 'Index-Normalized']])
print('Tesla:')
print(tesla_df[['Days', 'Close', 'Z-Normalized', 'Index-Normalized']])


plt.figure(figsize=(14, 7))
plt.plot(voest_df['Days'], voest_df['Z-Normalized'], label='Voestalpine', color='black')
plt.plot(tesla_df['Days'], tesla_df['Z-Normalized'], label='Tesla', color='red')
plt.title('Z-Normalized Stock Prices of Voestalpine vs. Tesla')
plt.xlabel('Days')
plt.ylabel('Z-Normalized Price')
plt.legend()
plt.show()

plt.figure(figsize=(14, 7))
plt.plot(voest_df['Days'], voest_df['Index-Normalized'], label='Voestalpine', color='black')
plt.plot(tesla_df['Days'], tesla_df['Index-Normalized'], label='Tesla', color='red')
plt.title('Index-Normalized Stock Prices of Voestalpine vs. Tesla')
plt.xlabel('Days')
plt.ylabel('Index-Normalized Price')
plt.legend()
plt.show()

#Correlation

corr_coefficient_voest = voest_df['Z-Normalized'].corr(voest_df['Days'])
covariance_voest = voest_df['Close'].cov(voest_df['Days'])

corr_coefficient_tesla = tesla_df['Z-Normalized'].corr(tesla_df['Days'])
covariance_tesla = tesla_df['Close'].cov(tesla_df['Days'])

corr_coefficient_voest_i = voest_df['Index-Normalized'].corr(voest_df['Days'])

corr_coefficient_tesla_i = tesla_df['Index-Normalized'].corr(tesla_df['Days'])


print("Z-Normalized for correlations:")
print("Pearson Correlation Coefficient for Voest:", corr_coefficient_voest)
print("Pearson Correlation Coefficient for Tesla:", corr_coefficient_tesla)

print("Index-Normalized for correlations:")
print("Pearson Correlation Coefficient for Voest:", corr_coefficient_voest_i)
print("Pearson Correlation Coefficient for Tesla:", corr_coefficient_tesla_i)

print("Covariance for Voest:", covariance_voest)
print("Covariance for Tesla:", covariance_tesla)

#For correlations:
#1 indicates a perfect positive linear relationship,
#0 indicates no linear relationship,
#-1 indicates a perfect negative linear relationship.
#That indicates that for Voest the stock price rose the later the date - it is a moderate positive correlation
#For Tesla the opposite is the case - it even has a slight negative correlation which means that the prices got lower the further the time
#There is no difference between the normalisations when it comes to correlations

#For covariance:
#The positive covariance indicates that both attributes move "into the same direction"
#While for Voest this is the case, for Tesla it's not


#pet_chaange() computes the percentage change from the previous row to the current row in the specified column.
voest_df['Daily Return'] = voest_df['Close'].pct_change()
tesla_df['Daily Return'] = tesla_df['Close'].pct_change()

# Calculate the rolling standard deviation of the stock price in a month
voest_df['Volatility'] = voest_df['Daily Return'].rolling(window=30).std()
tesla_df['Volatility'] = tesla_df['Daily Return'].rolling(window=30).std()

plt.figure(figsize=(14, 7))
plt.plot(voest_df['Days'], voest_df['Volatility'], label='Voestalpine Volatility')
plt.plot(tesla_df['Days'], tesla_df['Volatility'], label='Tesla Volatility', color='red')
plt.title('30-Day Rolling Volatility of Voestalpine and Tesla Stocks')
plt.xlabel('Day')
plt.ylabel('30-Day Rolling Standard Deviation of Daily Returns')
plt.legend()
plt.show()

#Volatility is the degree of variation of a trading price series over time
#many changes and high peaks --> lots of changes in the standard veviation and therefore the stock is less "stable"
#as well as the fluctation higher monthly
#Tesla shows a lower volitility and less peaks and therefore is generally "saver" to invest, than Voest which seems
#to have had more periods of high price fluctuations, especially recently
