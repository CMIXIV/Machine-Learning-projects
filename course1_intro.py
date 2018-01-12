#LESSON 1
# ** - exponentiation
#print(4 ** 2)
#ex how much is your 100$ worth after 10 years at 10% interest
#print(100*(1.1**7))

# Modulo
#print(18 % 7)

#variables
#height = 1.72
#weight = 69
#BMI=weight/height**2
#print(BMI)

#check the variable class
#print(type(BMI))
#integers (int), boolean (bool), float (float), string (str)
#you cannot check types of several variables at once via comma

#boolean variables are good for filtering operations
#z=True
#print(type(z))

#to sum numeric variables and strings, we need to convert numerics to strings first
#savings = 100
#result = 100 * 1.10 ** 7
#print("I started with $" + str(savings) + " and now have $" + str(result) + ". Awesome!")

#lists are built in square brackets
#heights=[a,b,c]
#lists can contain any types of variables, as well as several types
#there can be lists inside the lists

#indexing in lists starts with 0 - this is called zero-based indexing
#it is possible to index values in the list from the end - negative numbers are used for this
#fam=[1,2,3,4,5]
#print(fam[1])
#print(fam[-1])
# slicing - selecting several elements from a list, by using a range and a colon, thus creating another list
#print(fam[2:4])
#print(fam[:3])
#print(fam[3:])
#you can change elements in lists by using same square brackets
#fam[1]=3
#print(fam[1])
#fam[:2]=[2,3]
#print(fam)
#you can add lists as well - this merges lists together
#fam1=fam+[6,7]
#print(len(fam1))
#to delete elements from a list, use del
#del(fam1[:3])
#print(fam1)
#if you use the equal sign to make a copy of a list, it only creates a reference to the list, not a new list,
#so when you make a change in a new list, it will also make a change in the old list
#for this not to happen, you should use the list() operator or to select all values from the old list explicitly
#fam2=fam
#fam2[1]=10
#print(fam)
#print(fam2)
#fam=[1,2,3,4,5]
#fam3=list(fam)
#fam3[1]=10
#print(fam)
#print(fam3)
#fam4=fam[:]
#fam4[1]=10
#print(fam)
#print(fam4)
# you can place several commands in the same line by using ; sign
# you can print several elements of list by placing [][]
#print(round(1.68,1)); print(round(1.68))
#help() and ? before function both revert information about the function
#help(round)
#?round
#complex() - make a complex number out of a real one and optional of an imaginary part
#sorted() - sorts items in a list. reverse=True for reverse order

#methods are functions that you can apply to specific types of objects, and they start with a dot, like .index("xxx")
#fam=[1,2,3,4,5]
#print(fam.index(3))
#print(fam.count(2))
#.count() and .index() work on both strings and lists
#brother='nick'
#print(brother.capitalize())
#print(brother.replace('n','R'))
#print(brother.upper())
#fam.append(6)
#print(fam)
#in addition to .append(), .remove() and .reverse() are useful list methods


#math is a great mathematical package with variables, like the following:
#import math
#if you import the whole package, you need to have a syntax like this package.method()
#r=0.43
#C = 2*math.pi*r
#print(C)
#or
#r = 192500
#from math import radians
#if you import only one method, just type it like method()
#dist=r*radians(12)

import numpy as np
#y=np.array([1,2,3])
#print(y)
#baseball = [180, 215, 210, 210, 188, 176, 209, 200]
#np_baseball=np.array(baseball)
#np_height_m=np.array(baseball)/100
#numpy arrays allow making calculations on whole arrays at once
#y=np.array([1,2,3])
#z=np.array([3,4,5])
#bmi=y*z
#print(bmi)
# if you enter values of different types to a numpy array, it will convert all of them into strings
#conditions convert numpy arrays into boolean arrays
#print(y>2)
#subsetting on condition can be done with square brackets:
#y1=y[y>1]
#print(y1)
# or like this:
#y2=y>1
#print(y[y2])
#when coercing ingers and booleans, numpy converts booleans to 1 and 0
#there can be several dimensions in np arrays
#np_2d=np.array([y,z])
#.shape indicates the number of rows and columns in the numpy array
#print(np_2d.shape)
#it is possible to retrieve a specific element from a numpy array by indicating its row and column indices
#print(np_2d[0][2])
#print(np_2d[0,2])
#or you can only filter columns
#print(np_2d[:,0:2])
#np_mat = np.array([[1, 2],
#                   [3, 4],
#                   [5, 6]])
#print(np_mat * 2)
#print(np_mat + np.array([10, 10]))
#print(np_mat + np_mat)
#arithmetic functions over columns of data work as methods in numpy arrays
#print(np.mean(np_mat[:,0]))
#print(np.median(np_mat[:,1]))
# the following function is to check the correlation coefficients of columns
#print(np.corrcoef(np_mat[:,0],np_mat[:,1]))
# and this one calculates the standard deviation
#print(np.std(np_mat[:,0]))
# making arrays of random numbers
#height=np.round(np.random.normal(1.75,0.20,5000),2)
#weight=np.round(np.random.normal(60.32,15,5000),2)
#np_city=np.column_stack((height,weight))
#print(np_city)
# creating arrays filtered by certain values
#np_positions=np.array(positions)
#np_heights=np.array(heights)
#gk_heights=np.array(np_heights[np_positions=='GK'])
#other_heights=np.array(np_heights[np_positions!='GK'])
#print("Median height of goalkeepers: " + str(np.median(gk_heights)))
#print("Median height of other players: " + str(np.median(other_heights)))
