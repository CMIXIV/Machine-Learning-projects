#filename='huck_finn.txt'
#first we need to open a connection to a file
#file=open(filename,mode='r')
#where mode 'r' stands for reading
#file=open(filename,mode='w')
#mode 'w' to be able to write into the file
#text=file.read()
#print(text)
#in the end, don't forget to close the connection
#file.close()
#'with' as a context manager
#with open ('huck_finn.txt','r') as file:
#    print(file.read())
#iPython (jupyter) command showing contents of the current directory
#! ls
#check if the file is closed:
#print(file.closed)

#Printing a text line by line:
# Read & print the first 3 lines
#with open('moby_dick.txt') as file:
#    print(file.readline())
#    print(file.readline())
#    print(file.readline())

#If a file contains only DELIMITED NUMERIC data, the best practice is to open it with numpy
#import numpy as np
#filename='MNIST.txt'
#data=np.loadtxt(filename,delimiter='')
#print(data)
# this is for tab-delimited files:
#delimiter='\t'

#if we know the file contains a header, we can skip it:
#data=np.loadtxt(filename,delimiter='',skiprows=1)

#you can select which columns to import
#data=np.loadtxt(filename,delimiter='',skiprows=1,usecols=[0,2])

#you can import data in a specific format
#data=np.loadtxt(filename,delimiter='',dtype=str)

#practice with different loadtxt options
# Assign filename: file
#file = 'seaslug.txt'
# Import file: data
#data = np.loadtxt(file, delimiter='\t', dtype=str)
# Print the first element of data
#print(data[0])
# Import data as floats and skip the first row: data_float
#data_float = np.loadtxt(file, delimiter='\t', dtype=float, skiprows=1)
# Print the 10th element of data_float
#print(data_float[9])
# Plot a scatterplot of the data
#plt.scatter(data_float[:, 0], data_float[:, 1])
#plt.xlabel('time (min.)')
#plt.ylabel('percentage of larvae')
#plt.show()

#how to import files that have a mix of numericals and strings in it:
#data = np.genfromtxt('titanic.csv', delimiter=',', names=True, dtype=None)
#names=True tells us that there is a header
#dtype=None asks the program to figure out the datatype by itself

#recfromcsv is a function similar to genfromtxt, only it considers dtype=None by default
# Assign the filename: file
#file = 'titanic.csv'
# Import file using np.recfromcsv: d
#d=np.recfromcsv(file)
# Print out first three entries of d
#print(d[:3])

#USING PANDAS TO OPEN FLATFILES is the bst practice
#import pandas as pd
#filename='winequality-redself.csv'
#data=pd.read_csv(filename)
#checking the first rows
#data.head()
#convert into a numpy array
#data_array=data.values

# Assign the filename: file
#file = 'digits.csv'
# Read the first 5 rows of the file into a DataFrame: data
#data=pd.read_csv(file,nrows=5,header=None)
# Build a numpy array from the DataFrame: data_array
#data_array=data.values
# Print the datatype of data_array to the shell
#print(type(data_array))

#Correcting corrupt elements in a file
# Assign filename: file
#file = 'titanic_corrupt.txt'
# Import file: data
#data = pd.read_csv(file, sep='\t', comment='#', na_values='Nothing')
# Print the head of the DataFrame
#print(data.head())

#Pickled files are those recoded into a binary system to be readable by Python
#import pickle
#with open ('pickled-fruit.pkl','rb') as file:
#    data=pickle.load(file)
#print(data)

#importing Excel files
#import pandas as pd
#file='urbanpop.xlsx'
#data=pd.ExcelFile(file)
#print(data.sheet_names)
#then you can load a specific sheet as a DataFrame, indicating either a name of the sheet
#df1=data.parse('1960-1966')
#or its index
#df1=data.parse(0)

#print files containing in a current directory
#import os
#wd = os.getcwd()
#files=os.listdir(wd)
#for fil in files:
#    print(fil)

#IMPORTING A PICKLED FILE
# Import pickle package
#import pickle
# Open pickle file and load data: d
#with open('data.pkl','rb') as file:
#    d = pickle.load(file)
# Print d
#print(d)
# Print datatype of d
#print(type(d))

#LOADING AN EXCEL FILE AND CHECKING THE SHEET NAMES
# Import pandas
#import pandas as pd
# Assign spreadsheet filename: file
#file = 'battledeath.xlsx'
# Load spreadsheet: xl
#xl = pd.ExcelFile(file)
# Print sheet names
#print(xl.sheet_names)

#SELECTING SHEETS IN AN EXCEL FILE
# Load a sheet into a DataFrame by name: df1
#df1 = xl.parse('2004')
# Print the head of the DataFrame df1
#print(df1.head())
# Load a sheet into a DataFrame by index: df2
#df2=xl.parse(0)
# Print the head of the DataFrame df2
#print(df2.head())

#SELECTING SPECIFIC ROWS AND COLUMNS
# Parse the first sheet and rename the columns: df1
#df1 = xl.parse(0, skiprows=[1], names=['Country','AAM due to War (2002)'])
# Print the head of the DataFrame df1
#print(df1.head())
# Parse the first column of the second sheet and rename the column: df2
#df2 = xl.parse(1, parse_cols=[0], skiprows=[1], names=['Country'])
# Print the head of the DataFrame df2
#print(df2.head())

#importing SAS datafiles
#import pandas as pd
#from sas7bdat import SAS7BDAT
#with SAS7BDAT('urbanpop.sas7bdat') as file:
#    df_sas=file.to_data_frame()

#PRACTICE IMPORTING SAS FILES
# Import sas7bdat package
#from sas7bdat import SAS7BDAT
# Save file to a DataFrame: df_sas
#with SAS7BDAT('sales.sas7bdat') as file:
#    df_sas=file.to_data_frame()
# Print head of DataFrame
#print(df_sas.head())
# Plot histogram of DataFrame features (pandas and pyplot already imported)
#pd.DataFrame.hist(df_sas[['P']])
#plt.ylabel('count')
#plt.show()

#importing Stata files
#import pandas as pd
#data=pd.read_stata('disarea.dta')
#print(data.head())

#PRACTICE IMPORTING STATA FILES
# Import pandas
#import pandas as pd
# Load Stata file into a pandas DataFrame: df
#df=pd.read_stata('disarea.dta')
# Print the head of the DataFrame df
#print(df.head())
# Plot histogram of one column of the DataFrame
#pd.DataFrame.hist(df[['disa10']])
#plt.xlabel('Extent of disease')
#plt.ylabel('Number of coutries')
#plt.show()

#Importing HDF5 files
#import h5py
#filename='L-L1_LOSC_4_V1-1126259446-32.hdf5'
#data=h5py.File(filename,'r')
#for key in data.keys():
#    print(key)
#for key in data['meta'].keys():
#    print(key)
#print(data['meta']['Description'].value,data['meta']['Detector'].value)

# Import packages
#import numpy as np
#import h5py
# Assign filename: file
#file='LIGO_data.hdf5'
# Load file: data
#data = h5py.File(file, 'r')
# Print the datatype of the loaded file
#print(type(data))
# Print the keys of the file
#for key in data.keys():
#    print(key)

#PRACTICING WITH THE LIGO HDF5 DATA
# Get the HDF5 group: group
#group=data['strain']
# Check out keys of group
#for key in group.keys():
#    print(key)
# Set variable equal to time series data: strain
#strain=data['strain']['Strain'].value
# Set number of time points to sample: num_samples
#num_samples=10000
# Set time vector
#time = np.arange(0, 1, 1/num_samples)
# Plot data
#plt.plot(time, strain[:num_samples])
#plt.xlabel('GPS Time (s)')
#plt.ylabel('strain')
#plt.show()

#Reading MATLAB files
#import scipy.io
#filename='workspace.mat'
#mat=scipy.io.loadmat(filename)
#print(type(mat))
#scipy.io.savemat()

# Import package
#import scipy.io
# Load MATLAB file: mat
#mat=scipy.io.loadmat('albeck_gene_expression.mat')
# Print the datatype type of mat
#print(type(mat))

# Print the keys of the MATLAB dictionary
#print(mat.keys())
# Print the type of the value corresponding to the key 'CYratioCyt'
#print(type(mat['CYratioCyt']))
# Print the shape of the value corresponding to the key 'CYratioCyt'
#print(np.shape(mat['CYratioCyt']))
# Subset the array and plot it
#data = mat['CYratioCyt'][25, 5:]
#fig = plt.figure()
#plt.plot(data)
#plt.xlabel('time (min.)')
#plt.ylabel('normalized fluorescence (measure of expression)')
#plt.show()
