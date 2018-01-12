#COMMON DATA QUALITY PROBLEMS:
#inconsistent column names - capital letters or bad symbols
#missing data
#outliers
#duplicate rows
#untidy
#need to process columns
#column names can signal unexpected data problems
#import pandas as pd
#df=pd.read_csv('airquality.csv')
#print(df.head())
#print(df.tail())
#print(df.columns)
#print(df.shape)
#print(df.info())

#EXPLORATORY DATA ANALYSIS
#we start with frequency counts
#print(df.Month.value_counts(dropna=False))
#or
#print(df['Month'].value_counts(dropna=False))
#the 'head' method also works for value counts cutoff
#print(df.Ozone.value_counts(dropna=False).head(10))

#DESCRIPTIVE STATS OF VARIABLES
#here, 50% (50th percentile) is the median
#print(df.describe())

#use bar plots for discrete data visualization, and histogram for continuous
#import matplotlib.pyplot as plt
#%matplotlib inline
#df.Ozone.plot('hist')
#plt.show()

#to zoom into an issue, slice the data with bracket notations
#print(df[df.Ozone>100])

#boxplots is a good way of visualizing the shape of data
#df.boxplot(column='Ozone',by='Month')
#plt.show()

# Import necessary modules
#import pandas as pd
#import matplotlib.pyplot as plt
# Create the boxplot
#df.boxplot(column='initial_cost', by='Borough', rot=90)
# Display the plot
#plt.show()

# Import necessary modules
#import pandas as pd
#import matplotlib.pyplot as plt
# Create the boxplot
#df.boxplot(column='initial_cost', by='Borough', rot=90)
# Display the plot
#plt.show()

#when you have extremely high or extremely low values of outliers, it makes sense to log the scale
# Import matplotlib.pyplot
#import matplotlib.pyplot as plt
# Plot the histogram
#df['Ozone'].plot(kind='hist', rot=70, logx=True, logy=True)
# Display the histogram
#plt.show()

#SCATTERPLOTS are perfect for crossing 2 numeric variables
# Import necessary modules
#import pandas as pd
#import matplotlib.pyplot as plt
# Create and display the first scatter plot
#df.plot(kind='scatter', x='initial_cost', y='total_est_fee', rot=70)
#plt.show()
# Create and display the second scatter plot
#df_subset.plot(kind='scatter', x='initial_cost', y='total_est_fee', rot=70)
#plt.show()

#'Tidy Data' by Hadley Wickham, PhD

#COMBINING 2 COLUMNS INTO 1
#pd.melt(frame=df,id_vars='name',value_vars=['treatment a','treatment b'],var_name='treatment',value_name='result')

# Print the head of airquality
#print(airquality.head())
# Melt airquality: airquality_melt
#airquality_melt = pd.melt(frame=airquality, id_vars=['Month', 'Day'])
# Print the head of airquality_melt
#print(airquality_melt.head())

#with renaming of columns
# Print the head of airquality
#print(airquality.head())
# Melt airquality: airquality_melt
#airquality_melt = pd.melt(frame=airquality, id_vars=['Month', 'Day'], var_name='measurement', value_name='reading')
# Print the head of airquality_melt
#print(airquality_melt.head())

#PIVOTING is the opposite process to melting, when we create several variables out of one
#weather_tidy=weather.pivot(index='date',columns='element',values='value')

#when there are duplicate values for the same row to be, PIVOT TABLE with indication of an aggregation method is used
#weather2_tidy=weather.pivot(index='date',columns='element',values='value',aggfunc=np.mean)

# Print the head of airquality_melt
#print(airquality_melt.head())
# Pivot airquality_melt: airquality_pivot
#airquality_pivot = airquality_melt.pivot_table(index=['Month', 'Day'], columns='measurement', values='reading')
# Print the head of airquality_pivot
#print(airquality_pivot.head())

#Sometimes index resetting is needed
# Print the index of airquality_pivot
#print(airquality_pivot.index)
# Reset the index of airquality_pivot: airquality_pivot
#airquality_pivot = airquality_pivot.reset_index()
# Print the new index of airquality_pivot
#print(airquality_pivot.index)
# Print the head of airquality_pivot
#print(airquality_pivot.head())

# Pivot airquality_dup: airquality_pivot
#airquality_pivot = airquality_dup.pivot_table(index=['Month', 'Day'], columns='measurement', values='reading', aggfunc=np.mean)
# Reset the index of airquality_pivot
#airquality_pivot = airquality_pivot.reset_index()
# Print the head of airquality_pivot
#print(airquality_pivot.head())
# Print the head of airquality
#print(airquality.head())

#When several variables are in the same column, PARSING should be used to separate them. For example, we want to make the first letter in a string a separate variable
# Melt tb: tb_melt
#tb_melt = pd.melt(frame=tb, id_vars=['country', 'year'])
# Create the 'gender' column
#tb_melt['gender'] = tb_melt.variable.str[0]
# Create the 'age_group' column
#tb_melt['age_group'] = tb_melt.variable.str[1:]
# Print the head of tb_melt
#print(tb_melt.head())

#melt=pd.read_csv('ebola.csv')
# Melt ebola: ebola_melt
#ebola_melt = pd.melt(ebola, id_vars=['Date', 'Day'], var_name='type_country', value_name='counts')
# Create the 'str_split' column
#ebola_melt['str_split'] = ebola_melt['type_country'].str.split('_')
# Create the 'type' column
#ebola_melt['type'] = ebola_melt['str_split'].str.get(0)
# Create the 'country' column
#ebola_melt['country'] = ebola_melt['str_split'].str.get(1)
# Print the head of ebola_melt
#print(ebola_melt.head())

#DATA CONCATENATION
#con=pd.concat([weather_p1,weather_p2])
#then it is possible to select rows by index
#con=con.loc[0,:]
#but it is not good to have repetitive indices in the concatenated dataset. better re-index
#con=pd.concat([weather_p1,weather_p2],ignore_index=True)

#CONCATENATION IN ACTION
# Concatenate uber1, uber2, and uber3: row_concat
#row_concat = pd.concat([uber1,uber2,uber3])
# Print the shape of row_concat
#print(row_concat.shape)
# Print the head of row_concat
#print(row_concat.head())

#ATTACHING NEW COLUMNS TO THE DATA
# Concatenate ebola_melt and status_country column-wise: ebola_tidy
#ebola_tidy = pd.concat([ebola_melt,status_country],axis=1)
# Print the shape of ebola_tidy
#print(ebola_tidy.shape)
# Print the head of ebola_tidy
#print(ebola_tidy.head())

#there are also wildcards for finding multiple similar files based on a pattern
#e.g. *.csv stands for any file of csv format
#similarly, file_?.csv will retrieve any file that starts with file_ and then any single character

# Import necessary modules
#import pandas as pd
#import glob
# Write the pattern: pattern
#pattern = '*.csv'
# Save all file matches: csv_files
#csv_files = glob.glob(pattern)
# Print the file names
#print(csv_files)
# Load the second file into a DataFrame: csv2
#csv2 = pd.read_csv(csv_files[1])
# Print the head of csv2
#print(csv2.head())

#CONCATENATING MULTIPLES FILES INTO A SINGLE DATAFRAME
#import glob
#let's look for all csv files available
#csv_files=glob.glob('*.csv')
#print(csv_files)
#list_data=[]
#for filename in csv_files:
#    data=pd.read_csv(filename)
#    list_data.append(data)
#pd.concat(list_data)

# Create an empty list: frames
#frames = []
#  Iterate over csv_files
#for csv in csv_files:
    #  Read csv into a DataFrame: df
#    df = pd.read_csv(csv)
    # Append df to frames
#    frames.append(df)
# Concatenate frames into a single DataFrame: uber
#uber = pd.concat(frames)
# Print the shape of uber
#print(uber.shape)
# Print the head of uber
#print(uber.head())

#MERGING DATAFRAMES when the order of rows is not the same, and there are colunms in each dataset that are keys to merge upon
#pd.merge(left=state_populations,right=state_codes,on=None,left_on='state',right_on='name')
#if the columns to merge on have the same names in different datasets, we can use 'on' function
#types of merges: one-to-one, many-to-one/one-to-many, many-to-many

# Merge the DataFrames: o2o
#o2o = pd.merge(left=site, right=visited, left_on='name', right_on='site')
# Print o2o
#print(o2o)

#MANY-TO-MANY MERGE
# Merge site and visited: m2m
#m2m = pd.merge(left=site, right=visited, left_on='name', right_on='site')
# Merge m2m and survey: m2m
#m2m = pd.merge(left=m2m, right=survey, left_on='ident', right_on='taken')
# Print the first 20 lines of m2m
#print(m2m.head(20))

#CONVERTING DATA TYPES
#df['treatment b']=df['treatment b'].astype(str)
#df['sex']=df['sex'].astype('category')
#df['treatment a']=pd.to_numeric(df['treatment a'],errors='coerce')

# Convert the sex column to type 'category'
#tips.sex = tips.sex.astype('category')
# Convert the smoker column to type 'category'
#tips.smoker = tips.smoker.astype('category')
# Print the info of tips
#print(tips.info())

# Convert 'total_bill' to a numeric dtype
#tips['total_bill'] = pd.to_numeric(tips['total_bill'], errors='coerce')
# Convert 'tip' to a numeric dtype
#tips['tip'] = pd.to_numeric(tips['tip'], errors='coerce')
# Print the info of tips
#print(tips.info())

#REGULAR EXPRESSIONS as a formal way of specifying patterns in strings
#\d* - any digit, any number of times
#\$ - skip the dollar sign (or any sign)
#\d{2} - matches 2 digits next to each other
#^ - start macthing at the beginning of the value
#$ - match at the end of the value
#w - matches any alphanumeric character
#[A-Z] - matches any capital letter

#import re
#pattern=re.compile('\$\d*\.\d{2}')
#result=pattern.match('$17.89')
#print(bool(result))

# Import the regular expression module
#import re
# Compile the pattern: prog
#prog = re.compile('\d{3}-\d{3}-\d{4}')
# See if the pattern matches
#result = prog.match('123-456-7890')
#print(bool(result))
# See if the pattern matches
#result = prog.match('1123-456-7890')
#print(bool(result))

#FIND ALL MATCHES IN A STRING WITH RE.FINDALL()
# Import the regular expression module
#import re
# Find the numeric values: matches
#matches = re.findall('\d+', 'the recipe calls for 10 strawberries and 1 banana')
# Print the matches
#print(matches)

# Write the first pattern
#pattern1 = bool(re.match(pattern='\d{3}-\d{3}-\d{4}', string='123-456-7890'))
#print(pattern1)
# Write the second pattern
#pattern2 = bool(re.match(pattern='\$\d*\.\d{2}', string='$123.45'))
#print(pattern2)
# Write the third pattern
#pattern3 = bool(re.match(pattern='[A-Z]\w*', string='Australia'))
#print(pattern3)

#FUNCTIONS FOR DATA CLEANING - APPLY
#df.apply(np.mean,axis=0) - #applying mean value across all columns
#df.apply(np.mean,axis=1) - #applying mean value across all rows

#import re
#from numpy import NaN
#pattern=re.compile('^\$\d*\.\d{2}$') - #this is an expression that matches a monetary value in dollars with 2 decimals
#def diff_money(row,pattern):
#    icost=row['Initial Cost']
#    tef=row['Total Est. Fee']
#    if bool(pattern.match(icost)) and bool(pattern.match(tef)):
#        icost=icost.replace("$","")
#        tef=tef.replace("$","")
#        icost=float(icost)
#        tef=float(tef)
#        return icost-tef
#    else:
#        return(NaN)
#df_subset['diff']=df_subset.apply(diff_money,axis=1,pattern=pattern)

#RECODING GENDER VARIABLE INTO 1 AND 0
#from numpy import NaN
# Define recode_sex()
#def recode_sex(sex_value):
    # Return 1 if sex_value is 'Male'
#    if sex_value == 'Male':
#        return 1
    # Return 0 if sex_value is 'Female'
#    elif sex_value == 'Female':
#        return 0
    # Return np.nan
#    else:
#        return np.nan
# Apply the function to the sex column
#tips['sex_recode'] = tips.sex.apply(recode_sex)
# Print the first five rows of tips
#print(tips.head())

#USING LAMBDA FUNCTION
#2 ways to remove $ sign from a string
# Write the lambda function using replace
#tips['total_dollar_replace'] = tips.total_dollar.apply(lambda x: x.replace('$', ''))
# Write the lambda function using regular expressions
#tips['total_dollar_re'] = tips.total_dollar.apply(lambda x: re.findall('\d+\.\d+', x)[0])
# Print the head of tips
#print(tips.head())

#DROPPING DUPLICATES
#df=df.drop_duplicates()

# Create the new DataFrame: tracks which is a subset of columns from billboard
#tracks = billboard[['year','artist','track','time']]
# Print info of tracks
#print(tracks.info())
# Drop the duplicates: tracks_no_duplicates
#tracks_no_duplicates = tracks.drop_duplicates()
# Print info of tracks
#print(tracks_no_duplicates.info())

#DEALING WITH MISSING DATA
#first of all we need to count missings
#tips_nan.info()
#then we can drop them - this way we get a reduced dataframe with only entries that do not have missing values
#tips_dropped=tips_nan.dropna()
#but if too many entries have missing values, this is not viable. we can fill NAs instead
#tips_nan['sex']=tips_nan['sex'].fillna('missing')
#tips_nan[['total_bill','size']]=tips_nan[['total_bill','size']].fillna(0)
#mean_value=tips_non['tip'].mean()
#tips_non['tip']=tips_nan['tip'].fillna(mean_value)

# Calculate the mean of the Ozone column: oz_mean
#oz_mean = airquality.Ozone.mean()
# Replace all the missing values in the Ozone column with the mean
#airquality['Ozone'] = airquality['Ozone'].fillna(oz_mean)
# Print the info of airquality
#print(airquality.info())

#ASSERT method to make sure we have applied all desirable changes to the data correctly
# Assert that there are no missing values
#assert ebola.notnull().all().all()
#or
#assert pd.notnull(ebola).all().all()
# Assert that all values are >= 0
#assert (ebola>=0).all().all()

#ALLTOGETHER
#import pandas as pd
#df=pd.read_csv('my_data.csv')
#print(df.head())
#print(df.info())
#print(df.columns)
#print(df.describe())
#print(df.value_counts())
#print(df.column.plot('hist'))
#assert(df.column_data>0).all()
# Import matplotlib.pyplot
#import matplotlib.pyplot as plt
# Create the scatter plot
#g1800s.plot(kind='scatter', x='1800', y='1899')
# Specify axis labels
#plt.xlabel('Life Expectancy by Country in 1800')
#plt.ylabel('Life Expectancy by Country in 1899')
# Specify axis limits
#plt.xlim(20, 55)
#plt.ylim(20, 55)
# Display the plot
#plt.show()
#def check_null_or_valid(row_data):
    """Function that takes a row of data,
    drops all missing values,
    and checks if all remaining values are greater than or equal to 0
    """
#    no_na = row_data.dropna()[1:-1]
#    numeric = pd.to_numeric(no_na)
#    ge0 = numeric >= 0
#    return ge0
# Check whether the first column is 'Life expectancy'
#assert g1800s.columns[0] == 'Life expectancy'
# Check whether the values in the row are valid
#assert g1800s.iloc[:, 1:].apply(check_null_or_valid, axis=1).all().all()
# Check that there is only one instance of each country
#assert g1800s['Life expectancy'].value_counts()[0] == 1
# Concatenate the DataFrames row-wise
#gapminder = pd.concat([g1800s,g1900s,g2000s])
# Print the shape of gapminder
#print(gapminder.shape)
# Print the head of gapminder
#print(gapminder.head())
# Melt gapminder: gapminder_melt
#gapminder_melt = pd.melt(frame=gapminder,id_vars=['Life expectancy'])
# Rename the columns
#gapminder_melt.columns = ['country','year','life_expectancy']
# Print the head of gapminder_melt
#print(gapminder_melt.head())
# Convert the year column to numeric
#gapminder.year = pd.to_numeric(gapminder['year'], errors='coerce')
# Test if country is of type object
#assert gapminder.country.dtypes == np.object
# Test if year is of type int64
#assert gapminder.year.dtypes == np.int64
# Test if life_expectancy is of type float64
#assert gapminder.life_expectancy.dtypes == np.float64
# Convert the year column to numeric
#gapminder.year = pd.to_numeric(gapminder['year'], errors='coerce')
# Test if country is of type object
#assert gapminder.country.dtypes == np.object
# Test if year is of type int64
#assert gapminder.year.dtypes == np.int64
# Test if life_expectancy is of type float64
#assert gapminder.life_expectancy.dtypes == np.float64
# Assert that country does not contain any missing values
#assert pd.notnull(gapminder.country).all()
# Assert that year does not contain any missing values
#assert pd.notnull(gapminder.year).all()
# Drop the missing values
#gapminder = gapminder.dropna()
# Print the shape of gapminder
#print(gapminder.shape)
# Add first subplot
#plt.subplot(2, 1, 1)
# Create a histogram of life_expectancy
#gapminder.life_expectancy.plot(kind='hist')
# Group gapminder: gapminder_agg
#gapminder_agg = gapminder.groupby('year')['life_expectancy'].mean()
# Print the head of gapminder_agg
#print(gapminder_agg.head())
# Print the tail of gapminder_agg
#print(gapminder_agg.tail())
# Add second subplot
#plt.subplot(2, 1, 2)
# Create a line plot of life expectancy per year
#gapminder_agg.plot()
# Add title and specify axis labels
#plt.title('Life expectancy over the years')
#plt.ylabel('Life expectancy')
#plt.xlabel('Year')
# Display the plots
#plt.tight_layout()
#plt.show()
# Save both DataFrames to csv files
#gapminder.to_csv('gapminder.csv')
#gapminder_agg.to_csv('gapminder_agg.csv')
