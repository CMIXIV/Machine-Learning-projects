import matplotlib.pyplot as plt
# let's draw a line
#%matplotlib inline
#year=[1950,1970,1990,2010]
#popul=[2.519,3.692,5.263,6.972]
#plt.plot(year,popul)
#plt.show()
# the last line demonstrates the chart

#and now a scatterplot
#%matplotlib inline
#year=[1950,1970,1990,2010]
#popul=[2.519,3.692,5.263,6.972]
#plt.scatter(year,popul)
#plt.show()
#we can also show the size of bubbles:
#plt.scatter(year,popul,s=gdp)

# changing the x-scale to logarithmic is a good practice to spot correlations
#plt.xscale('log')

#histograms where x variable and the number of bins are the 2 most important parameters
#plt.hist(popul,bins=3)
#plt.show()

#plt.clf() empties the chart

#customize your output
#plt.xlabel('Year')
#plt.ylabel('Population')
#plt.title('World population')
#plt.yticks([0,2,4,6,8],['0','2B','4B','6B','8B'])
#while labeling, it's also possible to have string variables in the brackets
#some other settings for the plot that you can put inside scatter() arguments:
#alpha=0.8 - indicates opacity, from 0 to 1
#c=col - it is possible to create a dictionary with colors
#you can add labels to specific data points
#plt.text(1550, 71, 'India')
#plt.text(5700, 80, 'China')
#plt.grid(True) - adds a grid to the chart
#plt.show()

#DICTIONARIES
#world={"afghanistan":30.55,"albania":2.77,"algeria":39.21}
#print(world["albania"])

#MATCHING KEY-VALUE PAIR BY INDEX METHOD
# Definition of countries and capital
#countries = ['spain', 'france', 'germany', 'norway']
#capitals = ['madrid', 'paris', 'berlin', 'oslo']
# Get index of 'germany': ind_ger
#ind_ger=countries.index('germany')
# Use ind_ger to print out capital of Germany
#print(capitals[ind_ger])

#CREATING A DICTINARY FROM 2 LISTS
# Definition of countries and capital
#countries = ['spain', 'france', 'germany', 'norway']
#capitals = ['madrid', 'paris', 'berlin', 'oslo']
# From string in countries and capitals, create dictionary europe
#europe={'spain':'madrid','france':'paris','germany':'berlin','norway':'oslo'}
# Print europe
#print(europe)

#LOOKING UP VALUES BY KEYS
# Definition of dictionary
#europe = {'spain':'madrid', 'france':'paris', 'germany':'berlin', 'norway':'oslo' }
# Print out the keys in europe
#print(europe.keys())
# Print out value that belongs to key 'norway'
#print(europe['norway'])

#lists cannot be used as dictionary keys, as the keys should be immutable, and lists are mutable

#adding values to a dictionary:
#europe['belarus']='minsk'

#check if a key is in the dictinary:
#print('belarus' in europe)

#updating values in the dictinary:
#europe['spain']='barcelona'

#deleting items from a dictionary by key:
#del(europe['belarus'])

#ADDING ITEMS TO LISTS
# Definition of dictionary
#europe = {'spain':'madrid', 'france':'paris', 'germany':'berlin', 'norway':'oslo' }
# Add italy to europe
#europe['italy']='rome'
# Print out italy in europe
#print('italy' in europe)
# Add poland to europe
#europe['poland']='warsaw'
# Print europe
#print(europe)

#UPDATING AND MODIFYING ITEMS IN LISTS
# Definition of dictionary
#europe = {'spain':'madrid', 'france':'paris', 'germany':'bonn','norway':'oslo', 'italy':'rome', 'poland':'warsaw','australia':'vienna' }
# Update capital of germany
#europe['germany']='berlin'
# Remove australia
#del(europe['australia'])
# Print europe
#print(europe)

#DICTIONARIES IN DICTIONARIES
# Dictionary of dictionaries
#europe = { 'spain': { 'capital':'madrid', 'population':46.77 },
#           'france': { 'capital':'paris', 'population':66.03 },
#           'germany': { 'capital':'berlin', 'population':80.62 },
#           'norway': { 'capital':'oslo', 'population':5.084 } }
# Print out the capital of France
#print(europe['france']['capital'])
# Create sub-dictionary data
#data={'capital':'rome','population':59.83}
# Add data to europe under key 'italy'
#europe['italy']={'capital':'rome','population':59.83}
# Print europe
#print(europe)

#PANDAS
#dict={
#    "country":["Brazil","Russia","India","China","South Africa"],
#    "capital":["Brazilia","Moscow","New Delhi","Beijing","Pretoria"],
#    "area":[8.516,17.1,3.286,9.597,1.221],
#    "population":[200.4,143.5,1252,1357,52.98]
#}
#import pandas as pd
#brics=pd.DataFrame(dict)
#brics.index=['BR','RU','IN','CH','SA']
#print(brics)

#importing data
#brics=pd.read_csv("brics.csv")

#telling pandas that the first column are indices:
#brics=pd.read_csv("brics.csv",index_col=0)

#CONVERT LISTS INTO A DICTIONARY INTO A DATAFRAME
# Pre-defined lists
#names = ['United States', 'Australia', 'Japan', 'India', 'Russia', 'Morocco', 'Egypt']
#dr =  [True, False, False, False, True, True, True]
#cpc = [809, 731, 588, 18, 200, 70, 45]
# Import pandas as pd
#import pandas as pd
# Create dictionary my_dict with three key:value pairs: my_dict
#my_dict={'country':names,'drives_right':dr,'cars_per_cap':cpc}
# Build a DataFrame cars from my_dict: cars
#cars=pd.DataFrame(my_dict)
# Print cars
#print(cars)

#ASSIGN AN INDEX COLUMN TO DF
#import pandas as pd
# Build cars DataFrame
#names = ['United States', 'Australia', 'Japan', 'India', 'Russia', 'Morocco', 'Egypt']
#dr =  [True, False, False, False, True, True, True]
#cpc = [809, 731, 588, 18, 200, 70, 45]
#dict = { 'country':names, 'drives_right':dr, 'cars_per_cap':cpc }
#cars = pd.DataFrame(dict)
#print(cars)
# Definition of row_labels
#row_labels = ['US', 'AUS', 'JAP', 'IN', 'RU', 'MOR', 'EG']
# Specify row labels of cars
#cars.index=[row_labels]
# Print cars again
#print(cars)

#selecting specific columns:
#print(brics["country"])

#retrieving a type of data in a column
#type(brics["country"])

#making a column a sub-DataFrame
#print(brics[["country"]])
#print(brics[["country","capital"]])

#select specific rows from a df
#print(brics[1:3])
#indexing of rows also starts from 0

#loc - label-based selection of data
#iloc - position-based selection of data
#print(brics.loc["RU"])
#print(brics.loc[["RU","IN","CH"]])
#selecting specific rows and columns:
#print(brics.loc[["RU","IN","CH"],['country','capital']])
#print(brics.loc[:,['country','capital']])
#now with index-based method
#print(brics.iloc[[1]])
#print(brics.iloc[[1,2,3]])
#print(brics.iloc[[1,2,3],[0,1]])
#print(brics.iloc[:,[0,1]])

#PRINT COLUMNS FROM A DF
# Import cars data
#import pandas as pd
#cars = pd.read_csv('cars.csv', index_col = 0)
# Print out country column as Pandas Series
#print(cars['country'])
# Print out country column as Pandas DataFrame
#print(cars[['country']])
# Print out DataFrame with country and drives_right columns
#print(cars[['country','drives_right']])

#PRINT ROWS FROM A DF (remember the 0 row is column names)
# Import cars data
#import pandas as pd
#cars = pd.read_csv('cars.csv', index_col = 0)
# Print out first 3 observations
#print(cars[0:3])
# Print out fourth, fifth and sixth observation
#print(cars[3:6])

#FINDING ROWS BY LOC AND ILOC METHODS
# Import cars data
#import pandas as pd
#cars = pd.read_csv('cars.csv', index_col = 0)
# Print out observation for Japan
#print(cars.loc[["JAP"]])
# Print out observations for Australia and Egypt
#print(cars.iloc[[1,6]])

#FINDING BOTH COLUMNS AND ROWS BY LOC AND ILOC METHODS
# Import cars data
#import pandas as pd
#cars = pd.read_csv('cars.csv', index_col = 0)
# Print out drives_right value of Morocco
#print(cars.loc[['MOR'],['drives_right']])
# Print sub-DataFrame
#print(cars.loc[['RU','MOR'],['country','drives_right']])
#print(cars.iloc[[4,5],[1,2]])

#COMPARISONS
#print(2<3)
#print(2==3)

#COMPARING DIFFERENT OBJECT TYPES
# Comparison of booleans
#print(True==False)
# Comparison of integers
#print(-5*15!=75)
# Comparison of strings
#print('pyscript'=='PyScript')
# Compare a boolean with an integer
#print(True==1)
# Comparison of integers
#x = -3 * 6
#print(x>=-10)
# Comparison of strings
#y = "test"
#print("test"<=y)
# Comparison of booleans
#print(True>False)

#COMPARING NUMPY ARRAYS
# Create arrays
#import numpy as np
#my_house = np.array([18.0, 20.0, 10.75, 9.50])
#your_house = np.array([14.0, 24.0, 14.25, 9.0])
# my_house greater than or equal to 18
#print(my_house>=18)
# my_house less than your_house
#print(my_house<your_house)

#boolean operators: AND, OR, NOT
#x=12
#print(x>5 and x<15)
#y=5
#print(y<7 or y>13)
#in the numpy arrays, operators logical_and(),logical_or() and logical_not() are used instead
#import numpy as np
#bmi=np.array([21.8,20.9,21.7,24.7,21.4])
#print(bmi>21)
#print(np.logical_and(bmi>21,bmi<24))
#print(bmi[np.logical_and(bmi>21,bmi<24)])

#COMPARING VARIABLES
# Define variables
#my_kitchen = 18.0
#your_kitchen = 14.0
# my_kitchen bigger than 10 and smaller than 18?
#print(my_kitchen>10 and my_kitchen<18)
# my_kitchen smaller than 14 or bigger than 17?
#print(my_kitchen<14 or my_kitchen>17)
# Double my_kitchen smaller than triple your_kitchen?
#print(my_kitchen*2<your_kitchen*3)

#COMPLEX BOOLEAN EXPRESSIONS
#x = 8
#y = 9
#print(not(not(x < 3) and not(y > 14 or y > 10)))

#COMPARISON IN NUMPY ARRAYS
# Create arrays
#import numpy as np
#my_house = np.array([18.0, 20.0, 10.75, 9.50])
#your_house = np.array([14.0, 24.0, 14.25, 9.0])
# my_house greater than 18.5 or smaller than 10
#print(np.logical_or(my_house>18.5, my_house<10))
# Both my_house and your_house smaller than 11
#print(np.logical_and(my_house<11, your_house<11))

#CONDITIONAL OPERATORS IF,ELIF,ELSE
#z=5
#if z%2==0:
#    print("checking... "+str(z))
#    print(str(z)+" is even")
#else:
#    print("checking... "+str(z))
#    print(str(z)+" is odd")

#x=int(input("Enter a number:"))
#if x%2==0:
#    print(str(x)+" is divisible by 2")
#elif x%3==0:
#    print(str(x)+" is divisible by 3")
#else:
#    print(str(x)+" is neither divisible by 2 nor by 3")

#CONDITIONAL FILTERING OF PANDAS ARRAYS
#dict={
#    "country":["Brazil","Russia","India","China","South Africa"],
#    "capital":["Brazilia","Moscow","New Delhi","Beijing","Pretoria"],
#    "area":[8.516,17.1,3.286,9.597,1.221],
#    "population":[200.4,143.5,1252,1357,52.98]
#}
#import pandas as pd
#brics=pd.DataFrame(dict)
#brics.index=['BR','RU','IN','CH','SA']
#print(brics["area"]>8)
#print the rows that fit a conditing set on a column - there are direct and indirect ways for it
#is_huge=brics["area"]>8
#print(brics[is_huge])
#print(brics[brics["area"]>8])
#!!!filter out rows by condition and save to a separate array
#import numpy as np
#print(brics[np.logical_and(brics["area"]>8,brics["area"]<10)])
#brics1=pd.DataFrame(brics[np.logical_and(brics["area"]>8,brics["area"]<10)])
#brics1.to_csv("brics1.csv")

#FILTER PANDAS CASES BY CREATING A CONDITIONAL VARIABLE
# Import cars data
#import pandas as pd
#cars = pd.read_csv('cars.csv', index_col = 0)
# Extract drives_right column as Series: dr
#dr=cars["drives_right"]==True
# Use dr to subset cars: sel
#sel=cars[dr]
# Print sel
#print(sel)

#FILTER PANDAS CASES BY A CONDITION IN SQUARE BRACKETS
# Import cars data
#import pandas as pd
#cars = pd.read_csv('cars.csv', index_col = 0)
# Convert code to a one-liner
#sel = cars[cars['drives_right']==True]
# Print sel
#print(sel)

#FILTERING PANDAS DATA A LONG WAY
# Import cars data
#import pandas as pd
#cars = pd.read_csv('cars.csv', index_col = 0)
# Create car_maniac: observations that have a cars_per_cap over 500
#cpc=cars["cars_per_cap"]
#many_cars=cpc>500
#car_maniac=cars[many_cars]
# Print car_maniac
#print(car_maniac)

#APPLYING SEVERAL CONDITIONS TO PANDAS DATA
# Import cars data
#import pandas as pd
#cars = pd.read_csv('cars.csv', index_col = 0)
# Import numpy, you'll need this
#import numpy as np
# Create medium: observations with cars_per_cap between 100 and 500
#cpc=cars["cars_per_cap"]
#between=np.logical_and(cpc>100,cpc<500)
#medium=cars[between]
# Print medium
#print(medium)

#WHILE LOOPS
#error=50
#while error>1:
#    error=error/2
#    print(error)

# Initialize offset
#offset=8
# Code the while loop
#while offset!=0:
#    print("correcting...")
#    offset=offset-1
#    print(offset)

#IF LOOPS INSIDE WHILE LOOPS
# Initialize offset
#offset = -6
# Code the while loop
#while offset != 0 :
#    print("correcting...")
#    if offset>0:
#        offset=offset-1
#    else:
#        offset=offset+1
#    print(offset)

#FOR LOOPS
#family=[1.73,1.68,1.71,1.89]
#for height in family:
#    print(height)

#you can enumerate items in a list in order to see their indices
#for index,height in enumerate(family):
#        print("index "+str(index)+": "+str(height))

#for-looping strings
#for c in "family":
#    print(c.capitalize())

# areas list
#areas = [11.25, 18.0, 20.0, 10.75, 9.50]
# Change for loop to use enumerate()
#for index,a in enumerate(areas) :
#    print("room "+str(index)+": "+str(a))

# areas list
#areas = [11.25, 18.0, 20.0, 10.75, 9.50]
# Code the for loop
#for index, area in enumerate(areas) :
#    print("room " + str(index+1) + ": " + str(area))

#ITERATING A LIST OF LISTS
# house list of lists
#house = [["hallway", 11.25],
#         ["kitchen", 18.0],
#         ["living room", 20.0],
#         ["bedroom", 10.75],
#         ["bathroom", 9.50]]
# Build a for loop from scratch
#for room, area in house:
#    print("the " + str(room) + " is " + str(area)+" sqm")

#ITERATING DICTIONARIES AND NUMPY ARRAYS
#world={"afghanistan":30.55,"albania":2.77,"algeria":39.21}
#for key,value in world.items():
#    print(str(key)+"---"+str(value))

# Definition of dictionary
#europe = {'spain':'madrid', 'france':'paris', 'germany':'bonn',
#          'norway':'oslo', 'italy':'rome', 'poland':'warsaw', 'australia':'vienna' }
# Iterate over europe
#for key,value in europe.items():
#    print("the capital of "+key+" is "+value)

#import numpy as np
#np_area=np.array([8.516,17.1,3.286,9.597,1.221])
#np_pop=np.array([200.4,143.5,1252,1357,52.98])
#stat=np.array([np_area,np_pop])
#for val in np.nditer(stat):
#    print(val)

# Import numpy as np
#import numpy as np
# For loop over np_height
#for height in np_height:
#    print(str(height)+" inches")
# For loop over np_baseball
#for elem in np.nditer(np_baseball):
#    print(elem)

#ITERATING PANDAS ARRAYS
dict={
    "country":["Brazil","Russia","India","China","South Africa"],
    "capital":["Brazilia","Moscow","New Delhi","Beijing","Pretoria"],
    "area":[8.516,17.1,3.286,9.597,1.221],
    "population":[200.4,143.5,1252,1357,52.98]
}
import pandas as pd
brics=pd.DataFrame(dict)
brics.index=['BR','RU','IN','CH','SA']

#for lab,row in brics.iterrows():
#    print(lab)
#    print(row)

# Import cars data
#import pandas as pd
#cars = pd.read_csv('cars.csv', index_col = 0)
# Iterate over rows of cars
#for ind,row in cars.iterrows():
#    print(ind)
#    print(row)

#for lab,row in brics.iterrows():
#    print(lab+": "+row['capital'])

# Import cars data
#import pandas as pd
#cars = pd.read_csv('cars.csv', index_col = 0)
# Adapt for loop
#for lab, row in cars.iterrows() :
#    print(lab+": "+str(row["cars_per_cap"]))

#calculating the length of strings in a pandas column
#for lab,row in brics.iterrows():
#    brics.loc[lab, "name_length"]=len(row["country"])
#print(brics["name_length"])

#ADDING A VARIABLE THAT IS A CAPITALIZED EXISTING VARIABLE
# Import cars data
#import pandas as pd
#cars = pd.read_csv('cars.csv', index_col = 0)
# Code for loop that adds COUNTRY column
#for lab,row in cars.iterrows():
#    cars.loc[lab,"COUNTRY"]=row["country"].upper()
# Print cars
#print(cars)

#calculating a new column quickly:
#brics["name_length"]=brics["country"].apply(len)
#print(brics["name_length"])

#CAPITALIZING A NEW COLUMN BY APPLY
# Import cars data
#import pandas as pd
#cars = pd.read_csv('cars.csv', index_col = 0)
# Use .apply(str.upper)
#cars["COUNTRY"] = cars["country"].apply(str.upper)
#print(cars)

#RANDOM NUMBERS GENERATOR
#import numpy as np
#print(np.random.rand())
#it's also possible to generate pseudo-random numbers that are repeatable, by tying them to seed
#np.random.seed(123)
#print(np.random.rand())
#np.random.seed(123)
#print(np.random.rand())

#toss either 0 or 1:
#np.random.seed(123)
#coin=np.random.randint(0,2)
#for i in range(20):
#    coin=np.random.randint(0,2)
#    if coin==0:
#        print('heads')
#    else:
#        print('tails')

#simulating a dice
# Import numpy and set seed
#import numpy as np
#np.random.seed(123)
# Use randint() to simulate a dice
#dice=np.random.randint(1,7)
#print(dice)
# Use randint() again
#dice=np.random.randint(1,7)
#print(dice)

#tackling the Empire State Building challenge
# Import numpy and set seed
#import numpy as np
#np.random.seed(123)
# Starting step
#step = 50
# Roll the dice
#dice=np.random.randint(1,7)
# Finish the control construct
#if dice <= 2 :
#    step = step - 1
#elif dice <=5 :
#    step=step+1
#else:
#    step = step + np.random.randint(1,7)
# Print out dice and step
#print(dice); print(step)

#saving random throws outcomes in a list
#import numpy as np
#np.random.seed(123)
#outcomes=[]
#for x in range(10):
#    coin=np.random.randint(0,2)
#    if coin==0:
#        outcomes.append('heads')
#    else:
#        outcomes.append('tails')
#print(outcomes)

#generating a random walk sequence
#import numpy as np
#np.random.seed(123)
#tails=[0]
#for x in range(10):
#    coin=np.random.randint(0,2)
#    tails.append(tails[x]+coin)
#print(tails)

#simulate a random walk up the ESB
# Import numpy and set seed
#import numpy as np
#np.random.seed(123)
# Initialize random_walk
#random_walk=[0]
# Complete the ___
#for x in range(100) :
    # Set step: last element in random_walk
#    step=random_walk[-1]
    # Roll the dice
#    dice = np.random.randint(1,7)
    # Determine next step
#    if dice <= 2:
#        step = max(0,step - 1)
#    elif dice <= 5:
#        step = step + 1
#    else:
#        step = step + np.random.randint(1,7)
    # append next_step to random_walk
#    random_walk.append(step)
# Print random_walk
#print(random_walk[-1])

#now make the plot of it
# Import matplotlib.pyplot as plt
#import matplotlib.pyplot as plt
# Plot random_walk
#plt.plot(random_walk)
# Show the plot
#plt.show()

#DISTRIBUTION
#import numpy as np
#import matplotlib.pyplot as plt
#np.random.seed(123)
#final_tails=[]
#for x in range(1000):
#    tails=[0]
#    for x in range(10):
#        coin=np.random.randint(0,2)
#        tails.append(tails[x]+coin)
#    final_tails.append(tails[-1])
#plt.hist(final_tails,bins=10)
#plt.show()

#now to the ESB ascent again
# Initialization
#import numpy as np
#np.random.seed(123)
# Initialize all_walks
#all_walks=[]
# Simulate random walk 10 times
#for i in range(10) :
    # Code from before
#    random_walk = [0]
#    for x in range(100) :
#        step = random_walk[-1]
#        dice = np.random.randint(1,7)
#        if dice <= 2:
#            step = max(0, step - 1)
#        elif dice <= 5:
#            step = step + 1
#        else:
#            step = step + np.random.randint(1,7)
#        random_walk.append(step)
    # Append random_walk to all_walks
#    all_walks.append(random_walk)
# Print all_walks
#print(all_walks)

#converting random walks into a numpy array and show, then transpose and show again
#import matplotlib.pyplot as plt
#import numpy as np
#np.random.seed(123)
#all_walks = []
#for i in range(10) :
#    random_walk = [0]
#    for x in range(100) :
#        step = random_walk[-1]
#        dice = np.random.randint(1,7)
#        if dice <= 2:
#            step = max(0, step - 1)
#        elif dice <= 5:
#            step = step + 1
#        else:
#            step = step + np.random.randint(1,7)
#        random_walk.append(step)
#    all_walks.append(random_walk)
# Convert all_walks to Numpy array: np_aw
#np_aw=np.array(all_walks)
# Plot np_aw and show
#plt.plot(np_aw)
#plt.show()
# Clear the figure
#plt.clf()
# Transpose np_aw: np_aw_t
#np_aw_t=np.transpose(np_aw)
# Plot np_aw_t and show
#plt.plot(np_aw)
#plt.show()

#inputting a chance of falling from the stairs to step 0:
#import matplotlib.pyplot as plt
#import numpy as np
#np.random.seed(123)
#all_walks = []
# Simulate random walk 250 times
#for i in range(250) :
#    random_walk = [0]
#    for x in range(100) :
#        step = random_walk[-1]
#        dice = np.random.randint(1,7)
#        if dice <= 2:
#            step = max(0, step - 1)
#        elif dice <= 5:
#            step = step + 1
#        else:
#            step = step + np.random.randint(1,7)
        # Implement clumsiness
#        if np.random.rand()<=0.001 :
#            step = 0
#        random_walk.append(step)
#    all_walks.append(random_walk)
# Create and plot np_aw_t
#np_aw_t = np.transpose(np.array(all_walks))
#plt.plot(np_aw_t)
#plt.show()

#plotting the distribution of stores reached:
#import matplotlib.pyplot as plt
#import numpy as np
#np.random.seed(123)
#all_walks = []
# Simulate random walk 500 times
#for i in range(500) :
#    random_walk = [0]
#    for x in range(100) :
#        step = random_walk[-1]
#        dice = np.random.randint(1,7)
#        if dice <= 2:
#            step = max(0, step - 1)
#        elif dice <= 5:
#            step = step + 1
#        else:
#            step = step + np.random.randint(1,7)
#        if np.random.rand() <= 0.001 :
#            step = 0
#        random_walk.append(step)
#    all_walks.append(random_walk)
# Create and plot np_aw_t
#np_aw_t = np.transpose(np.array(all_walks))
# Select last row from np_aw_t: ends
#ends=np_aw_t[-1]
# Plot histogram of ends, display plot
#plt.hist(ends)
#plt.show()
