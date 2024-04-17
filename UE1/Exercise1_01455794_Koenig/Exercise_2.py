import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data_1 = pd.read_csv('production_line1.csv')
data_2 = pd.read_csv('production_line2.csv')
data_3 = pd.read_csv('production_line3.csv')

#print(data_1.head())

#Task 1 - If the order has to be produced as fast as possible, which of the production lines do you choose? Why?

mean_data_1 = data_1['line1'].mean()
mean_data_2 = data_2['line2'].mean()
mean_data_3 = data_3['line3'].mean()

print(f'Line 1: {mean_data_1}')
print(f'Line 2: {mean_data_2}')
print(f'Line 3: {mean_data_3}')

fastest_line = min(mean_data_1, mean_data_2, mean_data_3)
if fastest_line == mean_data_1:
    print("Choose Line 1.")
elif fastest_line == mean_data_2:
    print("Choose Line 2.")
else:
    print("Choose Line 3.")

#for the plot

mean_times = {
    'Line 1': mean_data_1,
    'Line 2': mean_data_2,
    'Line 3': mean_data_3
}

mean_times_df = pd.DataFrame(list(mean_times.items()), columns=['Production Line', 'Average Production Time'])

print(mean_times_df.head())

plt.figure(figsize=(10, 6))
mean_times_df.plot(kind='bar', color='green')
plt.title('Average Time of the Lines')
plt.xticks(ticks=[0, 1, 2], labels=['Line 1', 'Line 2', 'Line 3'], rotation=45)
plt.ylabel('Average Time')

plt.figure(figsize=(10, 6))
sns.kdeplot(data_1['line1'], label='Line 1', bw_adjust=0.5)
sns.kdeplot(data_2['line2'], label='Line 2', bw_adjust=0.5)
sns.kdeplot(data_3['line3'], label='Line 3', bw_adjust=0.5)

plt.title('Distribution of Production Times per Unit')
plt.xlabel('Production Time (hours)')
plt.ylabel('Count')
plt.legend()
#plt.show()

#Task 2 -  If the order needs to be produced for just-in-time production, i.e. a reliable estimation of production time is necessary, which of the production lines do you choose? Why?

#For just-in-time (JIT) production, where a reliable estimation of production time is crucial, choosing the right production line involves considering factors beyond just the average production times. It's essential to assess the consistency and predictability of each production line's performance.

std_dev_data_1 = data_1['line1'].std()
std_dev_data_2 = data_2['line2'].std()
std_dev_data_3 = data_3['line3'].std()

mean_times_df['Standard Deviation'] = [std_dev_data_1, std_dev_data_2, std_dev_data_3]

print(mean_times_df)

global_min = min(min(data_1['line1']), min(data_2['line2']), min(data_3['line3']))
global_max = max(max(data_1['line1']), max(data_2['line2']), max(data_3['line3']))

plt.figure(figsize=(10, 6))
plt.subplot(1, 3, 1)  # 1 row, 1 columns, 1st subplot
plt.boxplot(data_1)
plt.title('Production Line 1')
plt.ylabel('Time (Hours)')
plt.ylim(global_min, global_max)
plt.tight_layout()

plt.subplot(1, 3, 2)  # 1 row, 1 columns, 1st subplot
plt.boxplot(data_2)
plt.title('Production Line 2')
plt.ylabel('Time (Hours)')
plt.ylim(global_min, global_max)
plt.tight_layout()

plt.subplot(1, 3, 3)  # 1 row, 1 columns, 1st subplot
plt.boxplot(data_3)
plt.title('Production Line 3')
plt.ylabel('Time (Hours)')
plt.ylim(global_min, global_max)
plt.tight_layout()

#plt.show()

#line 3 is in the just in time production the best to choose - it has the lowest variability and is really
#predicatable. It also has very little extremes to higher hours, while line 1 and line 2 do.

#Taks 4 -

#boxplot see above
# + see data like distribution, including median, quartiles, and outliers
# - does not show the distribution shape and density
# Usage: straightforward comparison of distributions

#violine
# + can also show the density of the distribution, as well as median and quartiles
# - can be more complex to be interpreted
# Usage: when understanding the distribution's shape and density is important  

df_line1 = data_1['line1'].to_frame(name='Production Time')
df_line2 = data_2['line2'].to_frame(name='Production Time')
df_line3 = data_3['line3'].to_frame(name='Production Time')


df_combined = pd.concat([
    pd.DataFrame({'Production Time': df_line1['Production Time'], 'Line': 'Line 1'}),
    pd.DataFrame({'Production Time': df_line2['Production Time'], 'Line': 'Line 2'}),
    pd.DataFrame({'Production Time': df_line3['Production Time'], 'Line': 'Line 3'})
], ignore_index=True)

plt.figure(figsize=(10, 6))
sns.violinplot(x='Line', y='Production Time', data=df_combined)

print(df_combined)

plt.title('Production Time Distribution by Line')
plt.xlabel('Production Line')
plt.ylabel('Production Time')
plt.show()