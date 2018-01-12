#use try-except cycle to show users of your function the limitations
#def sqrt(x):
#    try:
#        return x**0.5
#    except:
#        print('x must be an int or a float')
#n=float(input())
#print(sqrt(n))

#def sqrt(x):
#    try:
#        return x**0.5
#    except TypeError:
#        print('x must be an int or a float')
#n=float(input())
#print(sqrt(n))

#def sqrt(x):
#    if x<0:
#        raise ValueError ('x must be non-negative')
#    try:
#        return x**5
#    except TypeError:
#        print('x must be an int or a float')
#print(sqrt(-1))

# Define shout_echo
#def shout_echo(word1, echo=1):
#    """Concatenate echo copies of word1 and three
#    exclamation marks at the end of the string."""
    # Initialize empty strings: echo_word, shout_words
#    echo_word=str()
#    shout_words=str()
    # Add exception handling with try-except
#    try:
        # Concatenate echo copies of word1 using *: echo_word
#        echo_word = word1*echo
        # Concatenate '!!!' to echo_word: shout_words
#        shout_words = echo_word+'!!!'
#    except:
        # Print error message
#        print("word1 must be a string and echo must be an integer.")
    # Return shout_words
#    return shout_words
# Call shout_echo
#shout_echo("particle", echo="accelerator")

# Define shout_echo
#def shout_echo(word1, echo=1):
#    """Concatenate echo copies of word1 and three
#    exclamation marks at the end of the string."""
#    # Raise an error with raise
#    if echo<0:
#        raise ValueError('echo must be greater than 0')
    # Concatenate echo copies of word1 using *: echo_word
#    echo_word = word1 * echo
    # Concatenate '!!!' to echo_word: shout_word
#    shout_word = echo_word + '!!!'
    # Return shout_word
#    return shout_word
# Call shout_echo
#shout_echo("particle", echo=5)
