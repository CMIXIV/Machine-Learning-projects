#from urllib.request import urlretrieve
#url='http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
#urlretrieve(url,'winequality-white.csv')

#now with red wine
# Import package
#from urllib.request import urlretrieve
# Import pandas
#import pandas as pd
# Assign url of file: url
#url='https://s3.amazonaws.com/assets.datacamp.com/production/course_1606/datasets/winequality-red.csv'
# Save file locally
#urlretrieve(url,'winequality-red.csv')
# Read file into a DataFrame and print its head
#df = pd.read_csv('winequality-red.csv', sep=';')
#print(df.head())

# Import packages
#import matplotlib.pyplot as plt
#import pandas as pd
# Assign url of file: url
#url='https://s3.amazonaws.com/assets.datacamp.com/production/course_1606/datasets/winequality-red.csv'
# Read file into a DataFrame: df
#df=pd.read_csv(url,sep=';')
# Print the head of the DataFrame
#print(df.head())
# Plot first column of df
#pd.DataFrame.hist(df.ix[:, 0:1])
#plt.xlabel('fixed acidity (g(tartaric acid)/dm$^3$)')
#plt.ylabel('count')
#plt.show()

# Import package
#import pandas as pd
# Assign url of file: url
#url='http://s3.amazonaws.com/assets.datacamp.com/course/importing_data_into_r/latitude.xls'
# Read in all sheets of Excel file: xl
#xl=pd.read_excel(url,sheetname=None)
# Print the sheetnames to the shell
#print(xl.keys())
# Print the head of the first sheet (using its name, NOT its index)
#print(xl['1700'].head())

#retrieve data from web pages
#from urllib.request import urlopen,Request
#url='https://www.wikipedia.org'
#request=Request(url)
#response=urlopen(request)
#html=response.read()
#print(urlopen(Request(url='https://www.wikipedia.org')).read())
#urlopen(Request(url='https://www.wikipedia.org')).close()

# Import packages
#from urllib.request import urlopen,Request
# Specify the url
#url = "http://www.datacamp.com/teach/documentation"
# This packages the request: request
#request=Request(url)
# Sends the request and catches the response: response
#response=urlopen(request)
# Print the datatype of response
#print(type(response))
# Be polite and close the response!
#response.close()

# Import packages
#from urllib.request import urlopen, Request
# Specify the url
#url = "http://www.datacamp.com/teach/documentation"
# This packages the request
#request = Request(url)
# Sends the request and catches the response: response
#response=urlopen(request)
# Extract the response: html
#html=response.read()
# Print the html
#print(html)
# Be polite and close the response!
#response.close()

#requests package as the most popular python package
#import requests
#url='https://www.wikipedia.org'
#r=requests.get(url)
#text=r.text
#print(text)

# Import package
#import requests
# Specify the url: url
#url='http://www.datacamp.com/teach/documentation'
# Packages the request, send the request and catch the response: r
#r=requests.get(url)
# Extract the response: text
#text=r.text
# Print the html
#print(text)

#BEAUTIFUL SOUP
#from bs4 import BeautifulSoup
#import requests
#url='https://www.crummy.com/software/BeautifulSoup'
#r=requests.get(url)
#html_doc=r.text
#soup=BeautifulSoup(html_doc)
#soup.prettify()

#getting url resources as text
#print(soup.title)
#print(soup.get_text())
#for link in soup.find_all('a'):
#    print(link.get('href'))

# Import packages
#import requests
#from bs4 import BeautifulSoup
# Specify url: url
#url='https://www.python.org/~guido/'
# Package the request, send the request and catch the response: r
#r=requests.get(url)
# Extracts the response as html: html_doc
#html_doc=r.text
# Create a BeautifulSoup object from the HTML: soup
#soup=BeautifulSoup(html_doc)
# Prettify the BeautifulSoup object: pretty_soup
#pretty_soup=soup.prettify()
# Print the response
#print(pretty_soup)

#GETTING TITLE AND TEXT FROM A WEBSITE
# Import packages
#import requests
#from bs4 import BeautifulSoup
# Specify url: url
#url = 'https://www.python.org/~guido/'
# Package the request, send the request and catch the response: r
#r = requests.get(url)
# Extract the response as html: html_doc
#html_doc = r.text
# Create a BeautifulSoup object from the HTML: soup
#soup=BeautifulSoup(html_doc)
# Get the title of Guido's webpage: guido_title
#guido_title=soup.title
# Print the title of Guido's webpage to the shell
#print(guido_title)
# Get Guido's text: guido_text
#guido_text=soup.get_text()
# Print Guido's text to the shell
#print(guido_text)

#GETTING HTTP LINKS FROM A WEBSITE
# Import packages
#import requests
#from bs4 import BeautifulSoup
# Specify url
#url = 'https://www.python.org/~guido/'
# Package the request, send the request and catch the response: r
#r = requests.get(url)
# Extracts the response as html: html_doc
#html_doc = r.text
# create a BeautifulSoup object from the HTML: soup
#soup = BeautifulSoup(html_doc)
# Print the title of Guido's webpage
#print(soup.title)
# Find all 'a' tags (which define hyperlinks): a_tags
#a_tags=soup.find_all('a')
# Print the URLs to the shell
#for link in a_tags:
#    print(link.get('href'))

#APIS AND JSON
#import json
#with open('snakes.json','r') as json_file:
#    json_data=json.load(json_file)
#the data will be loaded as a dictionary
#for key,value in json_data.items():
#    print(key+':',value)

#iterate ke-value pairs by key
# Load JSON: json_data
#with open("a_movie.json") as json_file:
#    json_data = json.load(json_file)
# Print each key-value pair in json_data
#for k in json_data.keys():
#    print(k + ': ', json_data[k])

#Print values for specific keys
#with open("a_movie.json") as json_file:
#    json_data = json.load(json_file)
#print(json_data['Title'])
#print(json_data['Title'])

#CONNECTING TO IMDB API
#import requests
#url='http://www.omdbapi.com/?t=hackers'
#r=requests.get(url)
#now time to decode the json data
#json_data=r.json()
#for key,value in json_data.items():
#    print(key+':',value)

#PRINT OUT A TEXT RESPONSE
# Import requests package
#import requests
# Assign URL to variable: url
#url='http://www.omdbapi.com/?apikey=ff21610b&t=social+network'
# Package the request, send the request and catch the response: r
#r = requests.get(url)
# Print the text of the response
#print(r.text)

#PRINT OUT ALL ITEMS OF THE MOVIE INFO
# Import package
#import requests
# Assign URL to variable: url
#url = 'http://www.omdbapi.com/?apikey=ff21610b&t=social+network'
# Package the request, send the request and catch the response: r
#r = requests.get(url)
# Decode the JSON data into a dictionary: json_data
#json_data=r.json()
# Print each key-value pair in json_data
#for k in json_data.keys():
#    print(k + ': ', json_data[k])

#CALLING THE WIKIPEDIA API
# Import package
#import requests
# Assign URL to variable: url
#url = 'https://en.wikipedia.org/w/api.php?action=query&prop=extracts&format=json&exintro=&titles=pizza'
# Package the request, send the request and catch the response: r
#r = requests.get(url)
# Decode the JSON data into a dictionary: json_data
#json_data=r.json()
# Print the Wikipedia page extract
#pizza_extract = json_data['query']['pages']['24768']['extract']
#print(pizza_extract)

#TWITTER API
#import tweepy,json
#access_token= "935183983377305600-InyCCfIY4esRwkgd4I6CBTQMaG56UrI"
#access_token_secret="mKM42TNQYezckvpDGqrHxqIlV6slA75DFG3SxH60ENY4Q"
#consumer_key="jMiZ6yAleXUHfrjsgSqeKI5PD"
#consumer_secret="cvgDzZjud7eFySYCGUmuWiZXsiPZ0C9vIIONiUsOMFXCtNLhTr"
#auth=tweepy.OAuthHandler(consumer_key,consumer_secret)
#auth.set_access_token(access_token,access_token_secret)
#class MyStreamListener(tweepy.StreamListener):
#    def __init__(self, api=None):
#        super(MyStreamListener, self).__init__()
#        self.num_tweets = 0
#        self.file = open("tweets.txt", "w")
#    def on_status(self, status):
#        tweet = status._json
#        self.file.write( json.dumps(tweet) + '\n' )
#        self.num_tweets += 1
#        if self.num_tweets < 100:
#            return True
#        else:
#            return False
#        self.file.close()
#    def on_error(self, status):
#        print(status)
#l=MyStreamListener()
#stream=tweepy.Stream(auth,l)
#stream.filter(track=['apples','oranges'])

# Import package
#import tweepy
# Store OAuth authentication credentials in relevant variables
#access_token = "1092294848-aHN7DcRP9B4VMTQIhwqOYiB14YkW92fFO8k8EPy"
#access_token_secret = "X4dHmhPfaksHcQ7SCbmZa2oYBBVSD2g8uIHXsp5CTaksx"
#consumer_key = "nZ6EA0FxZ293SxGNg8g8aP0HM"
#consumer_secret = "fJGEodwe3KiKUnsYJC3VRndj7jevVvXbK2D5EiJ2nehafRgA6i"
# Pass OAuth details to tweepy's OAuth handler
#auth = tweepy.OAuthHandler(consumer_key,consumer_secret)
#auth.set_access_token(access_token,access_token_secret)

# Import package
import json
# String of path to file: tweets_data_path
tweets_data_path = 'tweets.txt'
# Initialize empty list to store tweets: tweets_data
tweets_data = []
# Open connection to file
tweets_file = open(tweets_data_path, "r")
# Read in tweets and store in list: tweets_data
for line in tweets_file:
    tweet = json.loads(line)
    tweets_data.append(tweet)
# Close connection to file
tweets_file.close()
# Import package
import pandas as pd
# Build DataFrame of tweet texts and languages
df = pd.DataFrame(tweets_data, columns=['text', 'lang'])
# Print head of DataFrame
print(df.head())
#create a class that counts words
import re
def word_in_text(word, tweet):
    word = word.lower()
    text = tweet.lower()
    match = re.search(word, tweet)
    if match:
        return True
    return False
# Initialize list to store tweet counts
[clinton, trump, sanders, cruz] = [0, 0, 0, 0]
# Iterate through df, counting the number of tweets in which
# each candidate is mentioned
for index, row in df.iterrows():
    clinton += word_in_text('clinton', row['text'])
    trump += word_in_text('trump', row['text'])
    sanders += word_in_text('sanders', row['text'])
    cruz += word_in_text('cruz', row['text'])
# Import packages
import matplotlib.pyplot as plt
import seaborn as sns
# Set seaborn style
sns.set(color_codes=True)
# Create a list of labels:cd
cd = ['clinton', 'trump', 'sanders', 'cruz']
# Plot histogram
ax = sns.barplot(cd, [clinton,trump,sanders,cruz])
ax.set(ylabel="count")
plt.show()
