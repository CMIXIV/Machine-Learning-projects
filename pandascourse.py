import pandas as pd

#CREATING A PANDAS DATAFRAME FROM LISTS
#cities=['Austin','Dallas','Austin']
#signups=[7,12,3]
#visitors=[139,237,326]
#weekdays=['Sun','Sun','Mon']
#list_labels=['cities','signups','visitors','weekdays']
#list_cols=[cities,signups,visitors,weekdays]
#zipped=list(zip(list_labels,list_cols))
#print(zipped)
#data=dict(zipped)
#users=pd.DataFrame(data)
#print(users)

# Zip the 2 lists together into one list of (key,value) tuples: zipped
#zipped=list(zip(list_keys,list_values))
# Inspect the list using print()
#print(zipped)
# Build a dictionary with the zipped list: data
#data = dict(zipped)
# Build and inspect a DataFrame from the dictionary: df
#df = pd.DataFrame(data)
#print(df)

#CONSTRUCTING A DATAFRAME OUT OF A DICTIONARY
#heights=[59.0,65.2,62.9]
#data={'height':heights,'sex'='M'}

# Make a string with the value 'PA': state
#state = 'PA'
# Construct a dictionary: data
#data = {'state':state, 'city':cities}
# Construct a DataFrame from dictionary data: df
#df = pd.DataFrame(data)
# Print the DataFrame
#print(df)

#IMPORTING DATA INTO A DATAFRAME
#sunspots=pd.read_csv('ISSN_D_tot.csv')

#WHEN THERE ARE NO COLUMN LABELS
#sunspots=pd.read_csv('ISSN_D_tot.csv',header=None)
#col_names=['year','month','day']
#sunspots=pd.read_csv('ISSN_D_tot.csv',header=None,names=col_names)

# Create a list of the new column labels: new_labels
#new_labels = ['year','population']
# Read in the file, specifying the header and names parameters: df2
#df2 = pd.read_csv('world_population.csv', header=0, names=new_labels)

# Split on the comma to create a list: column_labels_list
#column_labels_list = column_labels.split(',')
# Assign the new column labels to the DataFrame: df.columns
#df.columns = column_labels_list
# Remove the appropriate columns: df_dropped
#df_dropped = df.drop(list_to_drop,axis='columns')
# Print the output of df_dropped.head()
#print(df_dropped.head())

#print(type(df))
#print(df.shape)
#print(df.columns)
#print(df.index)
#slicing dataframes
#df.iloc[:5;:]
#df.iloc[-5:;:]
#print(df.head())
#print(df.head(3))
#print(df.tail())
#print(df.tail(3))
#print(df.info())

#VIEWING A SLICE OF DATA IN THE MIDDLE OF A DATAFRAME
#print(sunspots.iloc[100:120,:])

#SELECTING SPECIFIC COLUMNS TO WORK WITH
#sunspots=sunspots['sunspots','definite']

#BROADCASTING: you can assign specific values (e.g. nan) to every n-th row in a specified row
#df.iloc[::3,-1]=np.nan #- fill in every third row with 'nan' in the last column
#extracting a column into a separate dataframe
#column1=df['column1']
#you can ask pandas to write the column as numbers
#colnum=column1.values
#users['fees']=0
#print(users)

#CHANGING NAMES OF COLUMNS AND INDICES OF ROWS
#results.columns=['height','sex']
#results.index=[1,2,3,4,5]

# Build a list of labels: list_labels
#list_labels = ['year','artist','song','chart weeks']
# Assign the list of labels to the columns attribute: df.columns
#df.columns = list_labels

#ASSIGNING VALUES THAT SHOULD BE TREATED AS MISSING
#sunspots=pd.read_csv('ISSN_D_tot.csv',header=None,names=col_names,na_values=' -1')
#or we can do it only on a specific column
#sunspots=pd.read_csv('ISSN_D_tot.csv',header=None,names=col_names,na_values={'sunspots':['-1']})

#DATE-TIME MANIPULATIONS
#sunspots=pd.read_csv('ISSN_D_tot.csv',header=None,names=col_names,parse_dates=[[0,1,2]])
#this will create a new variable that can be later renamed
#sunspots.index=sunspots['year_month_day']
#sunspots.index.name='date'

#sales=pd.read_csv('sales-feb-2015.csv', parse_dates=True, index_col='Date')
#it is then possible to make sophisticated index-based selections of data
#print(sales.loc['2015-2-19 11:00:00','Company'])
#print(sales.loc['2015-2-5'])
#it does the same as the following formats
#print(sales.loc['February 5, 2015'])
#print(sales.loc['2015-Feb-5'])
#it is also possible to select by month and by year
#print(sales.loc['2015-2'])
#print(sales.loc['2015'])
#print(sales.loc['2015-2-16':'2015-2-20'])
#you can also reindex the date-time index, so as to have equal time intervals represented in the data
#evening_2_11=pd.to_datetime(['2015-2-11 20:00','2015-2-11 21:00','2015-2-11 22:00','2015-2-11 23:00'])
#sales.reindex(evening_2_11)

# Prepare a format string: time_format
#time_format = '%Y-%m-%d %H:%M'
# Convert date_list into a datetime object: my_datetimes
#my_datetimes = pd.to_datetime(date_list, format =time_format)
# Construct a pandas Series using temperature_list and my_datetimes: time_series
#time_series = pd.Series(temperature_list, index=my_datetimes)

# Extract the hour from 9pm to 10pm on '2010-10-11': ts1
#ts1 = ts0.loc['2010-10-11 21:00:00']
# Extract '2010-07-04' from ts0: ts2
#ts2 = ts0.loc['2010-07-04']
# Extract data from '2010-12-15' to '2010-12-31': ts3
#ts3 = ts0.loc['2010-12-15':'2010-12-31']

#EXTRACTING HOURS FROM THE DATATIME STRING
#sales['DATE'].dt.hour

#CONVERTING TIME INTO LOCAL
#central=sales['Date'].dt.tz_localize('US/Central').central.dt.convert('US/Eastern')

#FORWARD AND BACKWARD FILLING OF MISSINGS
#sales.reindex(evening_2_11, method='ffill')
#sales.reindex(evening_2_11, method='bfill')

# Reindex without fill method: ts3
#ts3 = ts2.reindex(ts1.index)
# Reindex with fill method, using forward fill: ts4
#ts4 = ts2.reindex(ts1.index,method='ffill')

#RESAMPLING
#hourly data downsampled to daily data
#daily_mean=sales.resample('D').mean()
#daily_totals=sales.resample('D').sum()
#downsampling to weekly frequencies
#weekly_counts=sales.resample('W').count()
#'min','T' - to monutes, 'H' - to hours, 'D' - to days, 'B' - to business days, 'W' - to weeks, 'M' - to months, 'Q' - to quarters, 'A' - to years
#it is also possible to multiply these expressions by integers
#sales.loc[:,'Units'].resample('2W').sum()
#upsampling can also be used to get finer time intervals
#two_days.resample('4H').ffill()

# Downsample to 6 hour data and aggregate by mean: df1
#df1 = df.Temperature.resample('6h').mean()
# Downsample to daily data and count the number of data points: df2
#df2 = df.Temperature.resample('D').count()

# Extract temperature data for August: august
#august = df.Temperature['2010-8']
# Downsample to obtain only the daily highest temperatures in August: august_highs
#august_highs = august.resample('D').max()
# Extract temperature data for February: february
#february = df.Temperature['2010-2']
# Downsample to obtain the daily lowest temperatures in February: february_lows
#february_lows = february.resample('D').min()

#INTERPOLATION
#when there are only decade data, and you need to have annual data, it is possible to fill in the between cells with a linear progression method
#population.resample('A').first().interpolate('linear')

#SMOOTHING
# Extract data from 2010-Aug-01 to 2010-Aug-15: unsmoothed
#unsmoothed = df['Temperature']['2010-8-1':'2010-8-15']
# Apply a rolling mean with a 24 hour window: smoothed
#smoothed = unsmoothed.rolling(window=24).mean()
# Create a new DataFrame with columns smoothed and unsmoothed: august
#august = pd.DataFrame({'smoothed':smoothed, 'unsmoothed':unsmoothed})
# Plot both smoothed and unsmoothed data using august.plot().
#august.plot()
#plt.show()

# Extract the August 2010 data: august
#august = df['Temperature']['2010-8']
# Resample to daily data, aggregating by max: daily_highs
#daily_highs = august.resample('D').max()
# Use a rolling 7-day window with method chaining to smooth the daily high temperatures in August
#daily_highs_smoothed = daily_highs.rolling(window=7).mean()
#print(daily_highs_smoothed)

#CLEANING A MESSY FILE
# Read the raw file as-is: df1
#df1 = pd.read_csv(file_messy)
# Print the output of df1.head()
#print(df1.head())
# Read in the file with the correct parameters: df2
#df2 = pd.read_csv(file_messy, delimiter=' ', header=3, comment='#')
# Print the output of df2.head()
#print(df2.head())

#LOGGING values in a dataframe with NumPy
# Import numpy
import numpy as np
# Create array of DataFrame values: np_vals
#np_vals = df.values
# Create new array of base 10 logarithm values: np_vals_log10
#np_vals_log10 = np.log10(np_vals)
# Create array of new DataFrame by passing df to np.log10(): df_log10
#df_log10 = np.log10(df)
# Print original and new data containers
#print(type(np_vals), type(np_vals_log10))
#print(type(df), type(df_log10))

#EXPORTING DATA
#sunspots.to_csv('sunspots.csv')
#sunspots.to_excel('sunspots.csv')

#DROPPING INDEX WHILE EXPORTING
# Save the cleaned up DataFrame to a CSV file without the index
#df2.to_csv(file_clean, index=False)
# Save the cleaned up DataFrame to an excel file without the index
#df2.to_excel('file_clean.xlsx', index=False)

#PLOTTING DATA
import matplotlib.pyplot as plt
#aapl=pd.read_csv('aapl.csv',index_col='date',parse_dates=True)
#close_series=aapl['close']
#close_series.plot()
#plt.show()
#another way to select (a) column(s) to plot
#column_list2 = ['Temperature (deg F)','Dew Point (deg F)']
#df[column_list2].plot()
#plt.show()
#plot all columns from the dataset - THE PREFERRABLE METHOD
#aapl.plot()
#or
#plt.plot(aapl) #-this way there will be no legend and no title
#plt.show()
#giving the plot a title
#plt.title('Apple stock prices')
#specifying the names of axes
#plt.xlabel('Hours since midnight August 1, 2010')
#plt.ylabel('Temperature (degrees F')
#indicating the sizes of plots
#df.plot(kind='scatter', x='hp', y='mpg', s=sizes)
#when the variables have very different scales, it's good idea to log them
#aapl.plot()
#plt.yscale('log')
#plt.show()
#you can also show different numering variables in different chart windows
#df.plot(subplots=True)
#plt.show()
#styling the charts
#aapl['open'].plot(color='b',style='.-',legend=True)
#aapl['open'].plot(color='r',style='.',legend=True)
#formatting the axis
#plt.axis('2001','2002',0,100)
#plot.show()
#selecting only specific data to show on the chart
#aapl.loc['2001':'2004',['open','close','high','low']].plot()
#saving the charts
#plt.savefig('aapl.png')
#plt.savefig('aapl.jpg')
#plt.savefig('aapl.pdf')
#plt.show()

# Create a plot with color='red'
#df.plot(color='red')
# Add a title
#plt.title('Temperature in Austin')
# Specify the x-axis label
#plt.xlabel('Hours since midnight August 1, 2010')
# Specify the y-axis label
#plt.ylabel('Temperature (degrees F)')
# Display the plot
#plt.show()

#EXPLORATORY DATA ANALYSIS (EDA)
#iris=pd.read_csv('iris.csv',index_col=0)
#print(iris.shape)
#iris.head()
#identifying interdependencies between variables
#iris.plot(x='sepal_length',y='sepal_width',kind='scatter')
#plt.show()
#making a boxplot
#iris.plot(y='sepal_length',kind='box')
#plt.show()
#making a histogram
#iris.plot(y='sepal_length',kind='hist')
#plt.show()
#histogram arguments - bins, range (min and max of bins), normed (normalized to 0-1), cumulative (compute cumulative distribution)
#iris.plot(y='sepal_length',kind='hist', bins=30, range=(4,8), normed=True, cumulative=True)
#plt.show()

# Make a list of the column names to be plotted: cols
#cols = ['weight','mpg']
# Generate the box plots
#df[cols].plot(kind='box',subplots=True)
# Display the plot
#plt.show()

#3 ways to plot charts that work for histograms, scatterplot, boxplot
#iris.plot(kind='hist')
#iris.plt.hist()
#iris.hist()

#BUILD 2 CHARTS ONE BELOW THE OTHER
# This formats the plots such that they appear on separate rows
#fig, axes = plt.subplots(nrows=2, ncols=1)
# Plot the PDF
#df.fraction.plot(ax=axes[0], kind='hist', bins=30, normed=True, range=(0,.3))
#plt.show()
# Plot the CDF
#df.fraction.plot(ax=axes[1], kind='hist', bins=30, normed=True, cumulative=True, range=(0,.3))
#plt.show()

# Display the box plots on 3 separate rows and 1 column
#fig, axes = plt.subplots(nrows=3, ncols=1)
# Generate a box plot of the fare prices for the First passenger class
#titanic.loc[titanic['pclass'] == 1].plot(ax=axes[0], y='fare', kind='box')
# Generate a box plot of the fare prices for the Second passenger class
#titanic.loc[titanic['pclass'] == 2].plot(ax=axes[1], y='fare', kind='box')
# Generate a box plot of the fare prices for the Third passenger class
#titanic.loc[titanic['pclass'] == 3].plot(ax=axes[2], y='fare', kind='box')
# Display the plot
#plt.show()

#STYLING PLOTS
#there are 3 characters that regulate style of plots: first is color (k=black), 2nd is market type (dot) and 3rd is line style (- for solid)
#sp500.loc['2012-4','Close'].plot(style='k.-')
#b - blue, g - green, r - red, c - cyan; o - circle, * - star, s - square, + - plus; : - dotted, _ - dashed
#area plots
#sp500['Close'].plot(kind='area')

#STATISTICAL DATA ANALYSIS
#print(df.describe())
#return the number of non-null entries in a given column
#iris['sepal_length'].count()
#iris.count()
#return the average of values of a numeric variable, ignoring null values
#iris['sepal_length'].mean()
#iris.mean()
#standard deviation
#iris.std()
#median
#iris.median()
#a median is a separate case of a quantile, i.e. 50th quantile
#iris.quantile(0.2)
#several quantiles can be ordered at once
#q=[0.25,0.75]
#iris.quantile(q)
#min and max
#iris.min()
#iris.max()

# Compute the global mean and global standard deviation: global_mean, global_std
#global_mean = df.mean()
#global_std = df.std()
# Filter the US population from the origin column: us
#us = df.loc[df['origin'] == 'US']
# Compute the US mean and US standard deviation: us_mean, us_std
#us_mean = us.mean()
#us_std = us.std()
# Print the differences
#print(us_mean - global_mean)
#print(us_std - global_std)

# Print summary statistics of the fare column with .describe()
#print(df['fare'].describe())
# Generate a box plot of the fare column
#df['fare'].plot(kind='box')
# Show the plot
#plt.show()

# Print the number of countries reported in 2015
#print(df['2015'].count())
# Print the 5th and 95th percentiles
#print(df['2015'].quantile([0.05,0.95]))
# Generate a box plot
#years = ['1800','1850','1900','1950','2000']
#df[years].plot(kind='box')
#plt.show()

# Reset the index of ts2 to ts1, and then use linear interpolation to fill in the NaNs: ts2_interp
#ts2_interp = ts2.reindex(ts1.index).interpolate(how='linear')
# Compute the absolute difference of ts1 and ts2_interp: differences
#differences = np.abs(ts1 - ts2_interp)
# Generate and print summary statistics of the differences
#print(differences.describe())

#DESCRIBING A CATEGORICAL variable
#iris['species'].describe()
#will yield the following stats: count as usual, unique - number of unique categories, top - most frequent category, freq - occurrences of top
#showing the list of values of a categorical variable
#iris['species'].unique()

#FILTERING
#indices=iris['species']=='setosa'
#setosa=iris.loc[indices,:]
#you then can delete an unnecessary column
#del setosa['species']
#or
#df[df['origin'] == 'US']

#MANIPULATING STRINGS
#making strings uppercase
#sales['Company'].str.upper()
#searching by a part of a word/string
#sales['Product'].str.contains('ware')
#this returns a boolean variable, and then 'True' lines containing the searched element, can be summed
#sales['Product'].str.contains('ware').sum()

# Strip extra whitespace from the column names: df.columns
#df.columns = df.columns.str.strip()
# Extract data for which the destination airport is Dallas: dallas
#dallas = df['Destination Airport'].str.contains('DAL')
# Compute the total number of Dallas departures each day: daily_departures
#daily_departures = dallas.resample('D').sum()
# Generate the summary statistics for daily Dallas departures: stats
#stats = daily_departures.describe()

# Buid a Boolean mask to filter out all the 'LAX' departure flights: mask
#mask = df['Destination Airport'] == 'LAX'
# Use the mask to subset the data: la
#la = df[mask]
# Combine two columns of data to create a datetime series: times_tz_none
#times_tz_none = pd.to_datetime(la['Date (MM/DD/YYYY)'] + ' ' + la['Wheels-off Time'])
# Localize the time to US/Central: times_tz_central
#times_tz_central = times_tz_none.dt.tz_localize('US/Central')
# Convert the datetimes from US/Central to US/Pacific
#times_tz_pacific = times_tz_central.dt.tz_convert('US/Pacific')

# Plot the raw data before setting the datetime index
#df.plot()
#plt.show()
# Convert the 'Date' column into a collection of datetime objects: df.Date
#df.Date = pd.to_datetime(df['Date'])
# Set the index to be the converted 'Date' column
#df.set_index('Date',inplace=True)
# Re-plot the DataFrame to see that the axis is now datetime aware!
#df.plot()
#plt.show()

# Convert the date column to string: df_dropped['date']
#df_dropped['date'] = df_dropped['date'].astype(str)
# Pad leading zeros to the Time column: df_dropped['Time']
#df_dropped['Time'] = df_dropped['Time'].apply(lambda x:'{:0>4}'.format(x))
# Concatenate the new date and Time columns: date_string
#date_string = df_dropped['date'] + df_dropped['Time']
# Convert the date_string Series to datetime: date_times
#date_times = pd.to_datetime(date_string, format='%Y%m%d%H%M')
# Set the index to be the new date_times container: df_clean
#df_clean = df_dropped.set_index(date_times)
# Print the output of df_clean.head()
#print(df_clean.head())

# Print the dry_bulb_faren temperature between 8 AM and 9 AM on June 20, 2011
#print(df_clean.loc['2011-6-20 8:00:00':'2011-6-20 9:00:00', 'dry_bulb_faren'])
# Convert the dry_bulb_faren column to numeric values: df_clean['dry_bulb_faren']
#df_clean['dry_bulb_faren'] = pd.to_numeric(df_clean['dry_bulb_faren'], errors='coerce')
# Print the transformed dry_bulb_faren temperature between 8 AM and 9 AM on June 20, 2011
#print(df_clean.loc['2011-6-20 8:00:00':'2011-6-20 9:00:00', 'dry_bulb_faren'])
# Convert the wind_speed and dew_point_faren columns to numeric values
#df_clean['wind_speed'] = pd.to_numeric(df_clean['wind_speed'], errors='coerce')
#df_clean['dew_point_faren'] = pd.to_numeric(df_clean['dew_point_faren'], errors='coerce')

# Print the median of the dry_bulb_faren column
#print(df_clean.dry_bulb_faren.median())
# Print the median of the dry_bulb_faren column for the time range '2011-Apr':'2011-Jun'
#print(df_clean.loc['2011-4':'2011-6', 'dry_bulb_faren'].median())
# Print the median of the dry_bulb_faren column for the month of January
#print(df_clean.loc['2011-1', 'dry_bulb_faren'].median())

# Downsample df_clean by day and aggregate by mean: daily_mean_2011
#daily_mean_2011 = df_clean.resample('D').mean()
# Extract the dry_bulb_faren column from daily_mean_2011 using .values: daily_temp_2011
#daily_temp_2011 = daily_mean_2011['dry_bulb_faren'].values
# Downsample df_climate by day and aggregate by mean: daily_climate
#daily_climate = df_climate.resample('D').mean()
# Extract the Temperature column from daily_climate using .reset_index(): daily_temp_climate
#daily_temp_climate = daily_climate.reset_index()['Temperature']
# Compute the difference between the two arrays and print the mean difference
#difference = daily_temp_2011 - daily_temp_climate
#print(difference.mean())

# Select days that are sunny: sunny
#sunny = df_clean.loc[df_clean['sky_condition']=='CLR']
# Select days that are overcast: overcast
#overcast = df_clean.loc[df_clean['sky_condition'].str.contains('OVC')]
# Resample sunny and overcast, aggregating by maximum daily temperature
#sunny_daily_max = sunny.resample('D').max()
#overcast_daily_max = overcast.resample('D').max()
# Print the difference between the mean of sunny_daily_max and overcast_daily_max
#print(sunny_daily_max.mean() - overcast_daily_max.mean())

# Import matplotlib.pyplot as plt
#import matplotlib.pyplot as plt
# Select the visibility and dry_bulb_faren columns and resample them: weekly_mean
#weekly_mean = df_clean[['visibility','dry_bulb_faren']].resample('W').mean()
# Print the output of weekly_mean.corr()
#print(weekly_mean.corr())
# Plot weekly_mean with subplots=True
#weekly_mean.plot(subplots=True)
#plt.show()

# Create a Boolean Series for sunny days: sunny
#sunny = df_clean['sky_condition'] == 'CLR'
# Resample the Boolean Series by day and compute the sum: sunny_hours
#sunny_hours = sunny.resample('D').sum()
# Resample the Boolean Series by day and compute the count: total_hours
#total_hours = sunny.resample('D').count()
# Divide sunny_hours by total_hours: sunny_fraction
#sunny_fraction = sunny_hours / total_hours
# Make a box plot of sunny_fraction
#sunny_fraction.plot(kind='box')
#plt.show()

# Resample dew_point_faren and dry_bulb_faren by Month, aggregating the maximum values: monthly_max
#monthly_max = df_clean[['dew_point_faren','dry_bulb_faren']].resample('M').max()
# Generate a histogram with bins=8, alpha=0.5, subplots=True
#monthly_max.plot(kind='hist',bins=8, alpha=0.5, subplots=True)
# Show the plot
#plt.show()

# Extract the maximum temperature in August 2010 from df_climate: august_max
#august_max = df_climate.loc['2010-Aug','Temperature'].max()
#print(august_max)
# Resample the August 2011 temperatures in df_clean by day and aggregate the maximum value: august_2011
#august_2011 = df_clean.loc['2011-Aug','dry_bulb_faren'].resample('D').max()
# Filter out days in august_2011 where the value exceeded august_max: august_2011_high
#august_2011_high = august_2011.loc[august_2011 > august_max]
# Construct a CDF of august_2011_high
#august_2011_high.plot(kind='hist', normed=True, cumulative=True, bins=25)
# Display the plot
#plt.show()
