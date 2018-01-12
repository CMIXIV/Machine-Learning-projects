#def square(value):
#    new_value=value**2
#    return new_value
#num=square(4)
#print(num)
#docstrings describe how the function is built and what it does
#def square(value):
#    """Return the square of a value"""
#    new_value=value**2
#    return new_value

# Define the function shout
#def shout():
#    """Print a string with three exclamation marks"""
    # Concatenate the strings: shout_word
#    shout_word='congratulations'+'!!!'
    # Print shout_word
#    print(shout_word)
# Call shout
#shout()

# Define shout with the parameter, word
#def shout(word):
#    """Print a string with three exclamation marks"""
    # Concatenate the strings: shout_word
#    shout_word = word + '!!!'
    # Print shout_word
#    print(shout_word)
# Call shout with the string 'congratulations'
#shout('congratulations')

# Define shout with the parameter, word
#def shout(word):
#    """Return a string with three exclamation marks"""
    # Concatenate the strings: shout_word
#    shout_word=word+'!!!'
    # Replace print with return
#    return(shout_word)
# Pass 'congratulations' to shout: yell
#yell=shout('congratulations')
# Print yell
#print(yell)

#def raise_to_power(value1,value2):
#    """Raise value1 to the power of value2"""
#    new_value=value1**value2
#    return new_value
#result=raise_to_power(2,3)
#print(result)

#tuples
#even_nums=(2,4,6)
#a,b,c=even_nums
#print(a);print(b);print(c)
#print(even_nums[1])

#def raise_both(value1,value2):
#    new_value1=value1**value2
#    new_value2=value2**value1
#    new_tuple=(new_value1,new_value2)
#    return(new_tuple)
#result=raise_both(2,4)
#print(result)

# Define shout with parameters word1 and word2
#def shout(word1, word2):
#    """Concatenate strings with three exclamation marks"""
    # Concatenate word1 with '!!!': shout1
#    shout1=word1+'!!!'
    # Concatenate word2 with '!!!': shout2
#    shout2=word2+'!!!'
    # Concatenate shout1 with shout2: new_shout
#    new_shout=shout1+shout2
    # Return new_shout
#    return new_shout
# Pass 'congratulations' and 'you' to shout(): yell
#yell=shout('congratulations','you')
# Print yell
#print(yell)

# Unpack nums into num1, num2, and num3
#num1, num2, num3 = nums
# Construct even_nums
#even_nums = (2, num2, num3)

# Define shout_all with parameters word1 and word2
#def shout_all(word1, word2):
    # Concatenate word1 with '!!!': shout1
#    shout1=word1+'!!!'
    # Concatenate word2 with '!!!': shout2
#    shout2=word2+'!!!'
    # Construct a tuple with shout1 and shout2: shout_words
#    shout_words=(shout1,shout2)
    # Return shout_words
#    return shout_words
# Pass 'congratulations' and 'you' to shout_all(): yell1, yell2
#word1='congratulations'
#word2='you'
#yell1,yell2=shout_all(word1,word2)
# Print yell1 and yell2
#print(yell1)
#print(yell2)

# Import pandas
#import pandas as pd
# Import Twitter data as DataFrame: df
#df = pd.read_csv('tweets.csv')
# Initialize an empty dictionary: langs_count
#langs_count = {}
# Extract column from DataFrame: col
#col = df['lang']
# Iterate over lang column in DataFrame
#for entry in col:
    # If the language is in langs_count, add 1
#    if entry in langs_count.keys():
#        langs_count[entry]+=1
    # Else add the language to langs_count, set the value to 1
#    else:
#        langs_count[entry]=1
# Print the populated dictionary
#print(langs_count)

# Define count_entries()
#def count_entries(df, col_name):
#    """Return a dictionary with counts of
#    occurrences as value for each key."""
    # Initialize an empty dictionary: langs_count
#    langs_count = {}
    # Extract column from DataFrame: col
#    col = df[col_name]
    # Iterate over lang column in DataFrame
#    for entry in col:
        # If the language is in langs_count, add 1
#        if entry in langs_count.keys():
#            langs_count[entry]+=1
        # Else add the language to langs_count, set the value to 1
#        else:
#            langs_count[entry]=1
    # Return the langs_count dictionary
#    return langs_count
# Call count_entries(): result
#result=count_entries(tweets_df,'lang')
# Print the result
#print(result)

#GLOBAL AND LOCAL VALUES
#local values are those defined within an object/class
#global values are those defined outside of a class
#new_val=10
#def square(value):
#    global new_val
#    new_val=new_val**2
#    return new_val
#print(square(10))
#print(square(200))
#print(square(3))

#HOW TO CHANGE THE GLOBAL VALUE OF A VARIABLE WITHIN A CLASS
# Create a string: team
#team = "teen titans"
# Define change_team()
#def change_team():
#    """Change the value of the global variable team."""
    # Use team in global scope
#    global team
    # Change the value of team in global: team
#    team="justice league"
# Print team
#print(team)
# Call change_team()
#change_team()
# Print team
#print(team)

#NESTED FUNCTIONS
#def mod2plus5(x1,x2,x3):
#    """Returns the remainder plus 5 of three values."""
#    def inner(x):
#        """Returns the remainder plus 5 of a value"""
#        return x%2+5
#    return(inner(x1),inner(x2),inner(x3))
#print(mod2plus5(100,100,100))

#def raise_val(n):
#    """return the inner function"""
#    def inner(x):
#        """raise x to the power of n"""
#        raised=x**n
#        return raised
#    return inner
#square=raise_val(2)
#print(square(5))

#use the word nonlocal to refer to all values of a function within a class
#sequence of scopes searched: local->enclosing->global->built-in

#NESTED FUNCTIONS PRACTICE
# Define three_shouts
#def three_shouts(word1, word2, word3):
#    """Returns a tuple of strings
#    concatenated with '!!!'."""
    # Define inner
#    def inner(word):
#        """Returns a string concatenated with '!!!'."""
#        return word + '!!!'
    # Return a tuple of strings
#    return (inner(word1), inner(word2), inner(word3))
# Call three_shouts() and print
#print(three_shouts('a', 'b', 'c'))

# Define echo
#def echo(n):
#    """Return the inner_echo function."""
    # Define inner_echo
#    def inner_echo(word1):
#        """Concatenate n copies of word1."""
#        echo_word = (word1) * n
#        return echo_word
    # Return inner_echo
#    return inner_echo
# Call echo: twice
#twice = echo(2)
# Call echo: thrice
#thrice = echo(3)
# Call twice() and thrice() then print
#print(twice('hello'), thrice('hello'))

# Define echo_shout()
#def echo_shout(word):
#    """Change the value of a nonlocal variable"""
    # Concatenate word with itself: echo_word
#    echo_word=word+word
    #Print echo_word
#    print(echo_word)
    # Define inner function shout()
#    def shout():
#        """Alter a variable in the enclosing scope"""
        #Use echo_word in nonlocal scope
#        nonlocal echo_word
        #Change echo_word to echo_word concatenated with '!!!'
#        echo_word = echo_word+'!!!'
    # Call function shout()
#    shout()
    #Print echo_word
#    print(echo_word)
#Call function echo_shout() with argument 'hello'
#echo_shout('hello ')

#FUNCTIONS WITH DEFAULT ARGUMENTS
#def power(number,pow=2):
#    """raise number to the power of pow"""
#    new_value=number**pow
#    return new_value
#print(power(9,3))
#print(power(9))

# Define shout_echo
#def shout_echo(word1, echo=1):
#    """Concatenate echo copies of word1 and three
#     exclamation marks at the end of the string."""
    # Concatenate echo copies of word1 using *: echo_word
#    echo_word = word1*echo
    # Concatenate '!!!' to echo_word: shout_word
#    shout_word = echo_word + '!!!'
    # Return shout_word
#    return shout_word
# Call shout_echo() with "Hey": no_echo
#no_echo = shout_echo('Hey')
# Call shout_echo() with "Hey" and echo=5: with_echo
#with_echo = shout_echo('Hey',echo=5)
# Print no_echo and with_echo
#print(no_echo)
#print(with_echo)

# Define shout_echo
#def shout_echo(word1, echo=1, intense=False):
#    """Concatenate echo copies of word1 and three
#    exclamation marks at the end of the string."""
    # Concatenate echo copies of word1 using *: echo_word
#    echo_word = word1 * echo
    # Capitalize echo_word if intense is True
#    if intense is True:
        # Capitalize and concatenate '!!!': echo_word_new
#        echo_word_new = echo_word.upper() + '!!!'
#    else:
        # Concatenate '!!!' to echo_word: echo_word_new
#        echo_word_new = echo_word + '!!!'
    # Return echo_word_new
#    return echo_word_new
# Call shout_echo() with "Hey", echo=5 and intense=True: with_big_echo
#with_big_echo = shout_echo('Hey',echo=5,intense=True)
# Call shout_echo() with "Hey" and intense=True: big_no_echo
#big_no_echo = shout_echo('Hey',intense=True)
# Print values
#print(with_big_echo)
#print(big_no_echo)

# Define count_entries()
#def count_entries(df, col_name='lang'):
#    """Return a dictionary with counts of
#    occurrences as value for each key."""
    # Initialize an empty dictionary: cols_count
#    cols_count = {}
    # Extract column from DataFrame: col
#    col = df[col_name]
    # Iterate over the column in DataFrame
#    for entry in col:
        # If entry is in cols_count, add 1
#        if entry in cols_count.keys():
#            cols_count[entry] += 1
        # Else add the entry to cols_count, set the value to 1
#        else:
#            cols_count[entry] = 1
    # Return the cols_count dictionary
#    return cols_count
# Call count_entries(): result1
#result1 = count_entries(tweets_df)
# Call count_entries(): result2
#result2 = count_entries(tweets_df,'source')
# Print result1 and result2
#print(result1)
#print(result2)

#FUNCTIONS WITH FLEXIBLE ARGUMENTS
#def add_all(*args):
#    """sums all values in *args together"""
    #initialize sum
#    sum_all=0
    #accumulate the sum
#    for num in args:
#        sum_all+=num
#    return sum_all
#print(add_all(1,2,3,4,5,6,7,8,9,10))

# Define gibberish
#def gibberish(*args):
#    """Concatenate strings in *args together."""
    # Initialize an empty string: hodgepodge
#    hodgepodge=str()
    # Concatenate the strings in args
#    for word in args:
#        hodgepodge += word
    # Return hodgepodge
#    return hodgepodge
# Call gibberish() with one string: one_word
#one_word = gibberish("luke")
# Call gibberish() with five strings: many_words
#many_words = gibberish("luke", "leia", "han", "obi", "darth")
# Print one_word and many_words
#print(one_word)
#print(many_words)

# Define count_entries()
#def count_entries(df, *args):
#    """Return a dictionary with counts of
#    occurrences as value for each key."""
    #Initialize an empty dictionary: cols_count
#    cols_count = {}
    # Iterate over column names in args
#    for col_name in args:
        # Extract column from DataFrame: col
#        col = df[col_name]
        # Iterate over the column in DataFrame
#        for entry in col:
            # If entry is in cols_count, add 1
#            if entry in cols_count.keys():
#                cols_count[entry] += 1
            # Else add the entry to cols_count, set the value to 1
#            else:
#                cols_count[entry] = 1
    # Return the cols_count dictionary
#    return cols_count
# Call count_entries(): result1
#result1 = count_entries(tweets_df, 'lang')
# Call count_entries(): result2
#result2 = count_entries(tweets_df, 'lang', 'source')
# Print result1 and result2
#print(result1)
#print(result2)

#FUNCTIONS WITH FLEXIBLE KEYWORD ARGUMENTS (FUNCTIONS PRECEDED BY IDENTIFIERS)
#def print_all(**kwargs):
#    """print out key-value pairs in kwargs"""
#    #print out the key-value pairs
#    for key,value in kwargs.items():
#        print(key+": "+value)
#print_all(name="dumbledore",job="wizard")

# Define report_status
#def report_status(**kwargs):
#    """Print out the status of a movie character."""
#    print("\nBEGIN: REPORT\n")
    # Iterate over the key-value pairs of kwargs
#    for key, value in kwargs.items():
        # Print out the keys and values, separated by a colon ':'
#        print(key + ": " + value)
#    print("\nEND REPORT")
# First call to report_status()
#report_status(name="luke",affiliation="jedi",status="missing")
# Second call to report_status()
#report_status(name="anakin", affiliation="sith lord", status="deceased")

#LAMBDA - A WAY TO WRITE FUNCTIONS ON THE FLY:
#raise_to_power=lambda x,y:x**y
#print(raise_to_power(2,3))

# Define echo_word as a lambda function: echo_word
#echo_word = lambda word1,echo:word1*echo
# Call echo_word: result
#result = echo_word('hey',5)
# Print result
#print(result)

#MAP() applies a function to all elements in a list
#nums=[47,354,67,23,76,453]
#square_all=map(lambda num:num**2,nums)
#print(list(square_all))

# Create a list of strings: spells
#spells = ["protego", "accio", "expecto patronum", "legilimens"]
# Use map() to apply a lambda function over spells: shout_spells
#shout_spells = map(lambda item:item+'!!!', spells)
# Convert shout_spells to a list: shout_spells_list
#shout_spells_list=list(shout_spells)
# Convert shout_spells into a list and print it
#print(shout_spells_list)

#FILTER() function applies a logical condition to a list and selects only elements that satisfy this condition
# Create a list of strings: fellowship
#fellowship = ['frodo', 'samwise', 'merry', 'aragorn', 'legolas', 'boromir', 'gimli']
# Use filter() to apply a lambda function over fellowship: result
#result = filter(lambda member:len(member)>6, fellowship)
# Convert result to a list: result_list
#result_list=list(result)
# Convert result into a list and print it
#print(result_list)

#FILTERING ENTRIES FROM A DATAFRAME
# Select retweets from the Twitter DataFrame: result
#result = filter(lambda x:x[0:2]=='RT', tweets_df['text'])
# Create list from filter object result: res_list
#res_list=list(result)
# Print all retweets in res_list
#for tweet in res_list:
#    print(tweet)

#REDUCE() function reduces a list of elements to a single element
# Import reduce from functools
#from functools import reduce
# Create a list of strings: stark
#stark = ['robb', 'sansa', 'arya', 'eddard', 'jon']
# Use reduce() to apply a lambda function over stark: result
#result = reduce(lambda item1, item2: item1 + item2, stark)
# Print the result
#print(result)
