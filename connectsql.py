#first of all we need to establish a connection to a database and see what's in it
#from sqlalchemy import create_engine
#engine=create_engine('sqlite:///Chinook.sqlite')
#table_names =engine.table_names()
#print(table_names)

#extracting data from SQL with the help of pandas
#from sqlalchemy import create_engine
#import pandas as pd
#engine=create_engine('sqlite:///Chinook.sqlite')
#con=engine.connect()
#rs=con.execute("SELECT*FROM Artist")
#a * sign here selects all columns
#df=pd.DataFrame(rs.fetchall())
#fetchall function selects all rows, the line puts them into a pandas dataframe. alternatively,
#df=pd.DataFrame(rs.fetchmany(3))
#df.columns=rs.keys()
#con.close
#print(df.head())

#you can also select individual columns via the Context Manager construct:
#from sqlalchemy import create_engine
#import pandas as pd
#engine=create_engine('sqlite:///Chinook.sqlite')
# Open engine in context manager
# Perform query and save results to DataFrame: df
#with engine.connect() as con:
#    rs = con.execute("SELECT LastName,Title FROM Employee")
#    df = pd.DataFrame(rs.fetchmany(3))
#    df.columns = rs.keys()
# Print the length of the DataFrame df
#print(len(df))
# Print the head of the DataFrame df
#print(df.head())

#WHERE is a filtering function for rows that can take both strings (names) and numbers (integers,floats)
#with engine.connect() as con:
#    rs = con.execute("SELECT * FROM Employee WHERE EmployeeId >= 6")
#    df = pd.DataFrame(rs.fetchall())
#    df.columns = rs.keys()
#print(df)

#Sorting rows in a column is easy with ORDER BY function
# Open engine in context manager
#with engine.connect() as con:
#    rs = con.execute("SELECT * FROM Employee ORDER BY BirthDate")
#    df = pd.DataFrame(rs.fetchall())
    # Set the DataFrame's column names
#    df.columns = rs.keys()
# Print head of DataFrame
#print(df.head())

#READING DATA DIRECTLY FROM SQL WITH PANDAS
#from sqlalchemy import create_engine
#import pandas as pd
#    engine=create_engine('sqlite:///Chinook.sqlite')
#df=pd.read_sql_query("SELECT * FROM Employee ORDER BY BirthDate",engine)

#Example of comparing 'Python' and 'pandas' methods
# Import packages
#from sqlalchemy import create_engine
#import pandas as pd
# Create engine: engine
#engine=create_engine('sqlite:///Chinook.sqlite')
# Execute query and store records in DataFrame: df
#df = pd.read_sql_query("SELECT*FROM Album", engine)
# Print head of DataFrame
#print(df.head())
# Open engine in context manager
# Perform query and save results to DataFrame: df1
#with engine.connect() as con:
#    rs = con.execute("SELECT * FROM Album")
#    df1 = pd.DataFrame(rs.fetchall())
#    df1.columns = rs.keys()
# Confirm that both methods yield the same result: does df = df1 ?
#print(df.equals(df1))

#a more complex quesry via pandas
# Import packages
#from sqlalchemy import create_engine
#import pandas as pd
# Create engine: engine
#engine=create_engine('sqlite:///Chinook.sqlite')
# Execute query and store records in DataFrame: df
#df = pd.read_sql_query("SELECT*FROM Employee WHERE EmployeeId>=6 ORDER BY BirthDate", engine)
# Print head of DataFrame
#print(df.head())

#how to extract data from several related tables by JOIN
#from sqlalchemy import create_engine
#import pandas as pd
# Create engine: engine
#engine=create_engine('sqlite:///Chinook.sqlite')
#df = pd.read_sql_query("SELECT Title, Name FROM Album INNER JOIN Artist on Album.ArtistID = Artist.ArtistID", engine)
#print(df.head())

#this is how you do the same with the context manager
# Open engine in context manager
# Perform query and save results to DataFrame: df
#with engine.connect() as con:
#    rs = con.execute("SELECT Title, Name FROM Album INNER JOIN Artist on Album.ArtistID = Artist.ArtistID")
#    df = pd.DataFrame(rs.fetchall())
#    df.columns = rs.keys()
# Print head of DataFrame df
#print(df.head())

#you can apply filtering to inner join as well
# Execute query and store records in DataFrame: df
#with engine.connect() as con:
#    rs = con.execute("SELECT * FROM PlaylistTrack INNER JOIN Track on PlaylistTrack.TrackId = Track.TrackId WHERE Milliseconds<250000")
#    df = pd.DataFrame(rs.fetchall())
#    df.columns = rs.keys()
# Print head of DataFrame
#print(df.head())

#or
# Execute query and store records in DataFrame: df
#df = pd.read_sql_query("SELECT * FROM PlaylistTrack INNER JOIN Track on PlaylistTrack.TrackId = Track.TrackId WHERE Milliseconds<250000", engine)
# Print head of DataFrame
#print(df.head())
