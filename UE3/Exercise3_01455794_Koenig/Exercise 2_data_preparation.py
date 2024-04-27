import pandas as pd


pd.set_option('display.max_columns', None)

flight_data = pd.read_csv('flight_data.csv')
weather_data = pd.read_csv('weather_data.csv', dtype={10: str})
airport_data = pd.read_csv('airport_timezones.csv')
timezones_data = pd.read_csv('timezones.csv')


total_rows_before = flight_data.shape[0]

#clean data

# Identify rows where 'cancelled' is 1
cancellation_rows = flight_data[flight_data['CANCELLED'] == 1].index

# Drop these rows from the dataframe
flight_data = flight_data.drop(cancellation_rows)

# Drop the 'cancellation' column from the dataframe
flight_data = flight_data.drop(columns=['CANCELLED'])
flight_data = flight_data.drop(columns=['CANCELLATION_CODE'])

missing_values_count = flight_data.isnull().sum()
missing_percentage = (missing_values_count / len(flight_data)) * 100
print("Missing Percentage before cleaning")
print(missing_percentage)


total_rows = flight_data.shape[0]

#print(flight_data.columns)

# List of columns to replace NaN values
columns_to_replace_nan = ['CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY']

# Replace NaN values with 0 in the specified columns
for column in columns_to_replace_nan:
    flight_data[column] = flight_data[column].fillna(0)


#drops all other rows with missing values
flight_data = flight_data.dropna()

# Define the placeholder values
placeholders = [666, 999]

contains_placeholder_flight = flight_data.isin(placeholders)
placeholder_rows_flight = flight_data[contains_placeholder_flight.any(axis=1)]
indices_to_drop_flight = []
for column in placeholder_rows_flight.columns:
    for index, value in placeholder_rows_flight[column].items():
        if value in placeholders:
            indices_to_drop_flight.append(index)
flight_data = flight_data.drop(indices_to_drop_flight)

total_rows_after = flight_data.shape[0]


print("Total number of rows flight_data: before and after", total_rows_before, total_rows_after)

# print(flight_data.head(10))

#weather data

total_rows_before = weather_data.shape[0]

missing_values_count = weather_data.isnull().sum()
missing_percentage = (missing_values_count / len(weather_data)) * 100
print("Missing Percentage before cleaning")
print(missing_percentage)
# #no missing data

contains_placeholder_weather = weather_data.isin(placeholders)
placeholder_rows_weather = weather_data[contains_placeholder_weather.any(axis=1)]
indices_to_drop_weather = []
for column in placeholder_rows_weather.columns:
    for index, value in placeholder_rows_weather[column].items():
        if value in placeholders:
            indices_to_drop_weather.append(index)
weather_data = weather_data.drop(indices_to_drop_weather)

total_rows_after = weather_data.shape[0]
print("Total number of rows weather data: before and after", total_rows_before, total_rows_after)

# timezones data

total_rows_before = timezones_data.shape[0]

missing_values_count = timezones_data.isnull().sum()
missing_percentage = (missing_values_count / len(timezones_data)) * 100
print("Missing Percentage before cleaning Timezones")
print(missing_percentage)
#No missing data

contains_placeholder_timezones = timezones_data.isin(placeholders)
placeholder_rows_timezones = timezones_data[contains_placeholder_timezones.any(axis=1)]
indices_to_drop_timezones = []
for column in placeholder_rows_weather.columns:
    for index, value in placeholder_rows_weather[column].items():
        if value in placeholders:
            indices_to_drop_timezones.append(index)
timezones_data = timezones_data.drop(indices_to_drop_timezones)


total_rows_after = timezones_data.shape[0]
print("Total number of rows timezones data: before and after", total_rows_before, total_rows_after)


# airport_timezones

total_rows_before = airport_data.shape[0]

missing_values_count = airport_data.isnull().sum()
missing_percentage = (missing_values_count / len(airport_data)) * 100
print("Missing Percentage before cleaning Airport")
print(missing_percentage)
#drop the rows with missing data
airport_data = airport_data.dropna()

contains_placeholder_airport = airport_data.isin(placeholders)
placeholder_rows_airport = airport_data[contains_placeholder_airport.any(axis=1)]
indices_to_drop_airport = []
for column in placeholder_rows_airport.columns:
    for index, value in placeholder_rows_airport[column].items():
        if value in placeholders:
            indices_to_drop_airport.append(index)
airport_data = airport_data.drop(indices_to_drop_airport)

total_rows_after = airport_data.shape[0]
print("Total number of rows airport data: before and after", total_rows_before, total_rows_after)


#Merge flight_data and airport_data

# print("Flight data:")
# print(flight_data.head(5))
#
# print("Weather data:")
# print(weather_data.head(5))

##help of chatgpt
# def clean_time(t):
#     if pd.isna(t):
#         return None  # Handle NaN values
#     t = str(t).strip()
#     if t.endswith('.'):
#         t = t[:-1] + '00'
#     else:
#         try:
#             t = f"{int(float(t)):04d}"
#         except ValueError:
#             return None
#     return t
#
##help of chatgpt
# def adjust_datetime(date, time):
#     """Adjusts the time and potentially the date if time is '2400', then rounds the time to 10 minutes."""
#     if pd.isna(time):
#         return date, None  # Handle NaN times
#
#     time = str(time).strip()
#     if time.endswith('.'):
#         time = time[:-1] + '00'  # Remove decimal and add '00'
#     elif time == '2400' or time == '2400.0':  # Handle '2400' and '2400.0'
#         # Convert date to datetime, increment by one day
#         new_date = pd.to_datetime(date) + pd.Timedelta(days=1)
#         time = '0000'  # Reset time to '0000'
#         date = new_date.strftime('%Y-%m-%d')
#     else:
#         try:
#             # Format time as four digits
#             time = f"{int(float(time)):04d}"
#         except ValueError:
#             return date, None  # Return None if conversion fails
#
#     # Combine date and time into a single datetime object
#     datetime_obj = pd.to_datetime(f"{date} {time}")
#
#     # Round to 10 minutes
#     datetime_rounded = datetime_obj.round('10min') # 10 minutes
#
#     # Return the adjusted date and the time as HHMM string
#     return datetime_rounded.strftime('%Y-%m-%d'), datetime_rounded.strftime('%H%M')
#
#
# # Apply cleaning and then adjustment to times and dates
# flight_data['WHEELS_OFF_R'] = flight_data['WHEELS_OFF'].apply(clean_time)
# flight_data['WHEELS_ON_R'] = flight_data['WHEELS_ON'].apply(clean_time)
# flight_data[['FL_DATE', 'WHEELS_OFF_R']] = flight_data.apply(lambda row: adjust_datetime(row['FL_DATE'], row['WHEELS_OFF_R']), axis=1, result_type='expand')
# flight_data[['FL_DATE', 'WHEELS_ON_R']] = flight_data.apply(lambda row: adjust_datetime(row['FL_DATE'], row['WHEELS_ON_R']), axis=1, result_type='expand')
#
# # Conversion to datetime
# flight_data['datetime_dep'] = pd.to_datetime(flight_data['FL_DATE'] + ' ' + flight_data['WHEELS_OFF_R'], errors='coerce')
# flight_data['datetime_arr'] = pd.to_datetime(flight_data['FL_DATE'] + ' ' + flight_data['WHEELS_ON_R'], errors='coerce')
# flight_data = flight_data.drop('WHEELS_OFF_R', axis=1)
# flight_data = flight_data.drop('WHEELS_ON_R', axis=1)
#
# #merge airport
#
# flight_data = flight_data.merge(airport_data, left_on='ORIGIN', right_on='iata_code', suffixes=('', '_dep'))
# flight_data = flight_data.merge(airport_data, left_on='DEST', right_on='iata_code', suffixes=('', '_arr'))
# flight_data.rename(columns={'iana_tz': 'iana_tz_dep', 'windows_tz': 'windows_tz_dep', 'iata_code': 'iata_code_dep'}, inplace=True)
# print("Checkpoint 1")
#
# #Merge timezones
#
# flight_data = flight_data.merge(timezones_data, left_on='iana_tz_dep', right_on='timezone', suffixes=('', '_dep'))
# flight_data = flight_data.merge(timezones_data, left_on='iana_tz_arr', right_on='timezone', suffixes=('', '_arr'))
# flight_data.rename(columns={'timezone': 'timezone_dep', 'offset': 'offset_dep', 'offset_dst': 'offset_dst_dep'}, inplace=True)
# print("Checkpoint 2")
#
# #Merge weather data
#
# weather_data = weather_data.drop(columns=['lon'])
# weather_data = weather_data.drop(columns=['lat'])
#
#
# weather_data['valid'] = pd.to_datetime(weather_data['valid'], errors='coerce')
#
#
# weather_data['valid_rounded'] = weather_data['valid'].dt.round('10min')
# print(weather_data['valid_rounded'].head(3))
#
#
# print("Flight data:")
# print(flight_data.head(5))
#
#
# flight_data = flight_data.merge(weather_data, left_on=['ORIGIN', 'datetime_dep'], right_on=['station', 'valid'],
#                                 suffixes=('', '_weather_dep'))
#
# flight_data.rename(columns={'station': 'station_dep', 'valid': 'valid_dep', 'tmpf': 'tmpf_dep', 'dwpf': 'dwpf_dep',
#                             'relh': 'relh_dep', 'drct': 'drct_dep', 'sknt': 'sknt_dep', 'p01i': 'p01i_dep',
#                             'alti': 'alti_dep', 'mslp': 'mslp_dep', 'vsby': 'vsby_dep', 'gust': 'gust_dep',
#                             'skyc1': 'skyc1_dep', 'skyc2': 'skyc2_dep', 'skyc3': 'skyc3_dep', 'skyc4': 'skyc4_dep',
#                             'skyl1': 'skyl1_dep', 'skyl2': 'skyl2_dep', 'skyl3': 'skyl3_dep', 'skyl4': 'skyl4_dep',
#                             'wxcodes': 'wxcodes_dep', 'ice_accretion_1hr': 'ice_accretion_1hr_dep', 'ice_accretion_3hr': 'ice_accretion_3hr_dep',
#                             'ice_accretion_6hr': 'ice_accretion_6hr_dep', 'peak_wind_gust': 'peak_wind_gust_dep', 'peak_wind_drct': 'peak_wind_drct_dep',
#                             'peak_wind_time': 'peak_wind_time_dep', 'feel': 'feel_dep'}, inplace=True)
#
#
# print("Flight data:")
# print(flight_data.head(5))
#
# total_rows = flight_data.shape[0]
# print(total_rows)
#
#
# # Save the merged data to a new csv file
# flight_data.to_csv('merged_data.csv', index=False)
