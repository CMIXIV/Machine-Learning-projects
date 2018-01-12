#square each element in a list
#nums=[12,8,3,4,6,8]
#new_nums=[num**2 for num in nums]
#print(new_nums)

#create a list which is a range
#result=[num for num in range(39,56,3)]
#print(result)

#list comprehension for nested loops
#pairs_2=[(num1,num2) for num1 in range(0,2) for num2 in range(6,8)]
#print(pairs_2)

#print out the first letters from the list of names:
#doctor=['house','cuddy','melon']
#print([doc[0] for doc in doctor])

# Create list comprehension: squares
#squares = [square**2 for square in range(10)]
#print(squares)

# Create a 5 x 5 matrix using a list of lists: matrix
#matrix = [[col for col in range(5)] for row in range(5)]
# Print the matrix
#for row in matrix:
#    print(row)

#CONDITIONALS IN COMPREHENSIONS
#print([num**2 for num in range(10) if num%2==0])

# Create a list of strings: fellowship
#fellowship = ['frodo', 'samwise', 'merry', 'aragorn', 'legolas', 'boromir', 'gimli']
# Create list comprehension: new_fellowship
#new_fellowship = [member for member in fellowship if len(member)>=7]
# Print the new list
#print(new_fellowship)

#print([num**2 if num%2==0 else 0 for num in range(10)])

# Create a list of strings: fellowship
#fellowship = ['frodo', 'samwise', 'merry', 'aragorn', 'legolas', 'boromir', 'gimli']
# Create list comprehension: new_fellowship
#new_fellowship = [member if len(member)>=7 else "" for member in fellowship]
# Print the new list
#print(new_fellowship)

#DICTIONARIES COMPREHENSIONS
#pos_neg={num: -num for num in range(10)}
#print(pos_neg)

# Create a list of strings: fellowship
#fellowship = ['frodo', 'samwise', 'merry', 'aragorn', 'legolas', 'boromir', 'gimli']
# Create dict comprehension: new_fellowship
#new_fellowship = {member: len(member) for member in fellowship}
# Print the new list
#print(new_fellowship)

#GENERATORS
#result=(num for num in range(6))
#for num in result:
#    print(num)
#print(list(result))

# Create generator object: result
#result = (num for num in range(31))
# Print the first 5 values
#print(next(result))
#print(next(result))
#print(next(result))
#print(next(result))
#print(next(result))
# Print the rest of the values
#for value in result:
#    print(value)

# Create a list of strings: lannister
#lannister = ['cersei', 'jaime', 'tywin', 'tyrion', 'joffrey']
# Create a generator object: lengths
#lengths = (len(person) for person in lannister)
# Iterate over and print the values in lengths
#for value in lengths:
#    print(value)

#you can do the same operations with generators as you can with list comprehensions
#even_nums=(num for num in range(10) if num%2==0)
#print(list(even_nums))

#generator functions create an iterable that can be further called:
#def num_sequence(n):
#    i=0
#    while i<n:
#        yield i
#        i+=1
#r=num_sequence(6)
#print(list(r))

# Create a list of strings
#lannister = ['cersei', 'jaime', 'tywin', 'tyrion', 'joffrey']
# Define generator function get_lengths
#def get_lengths(input_list):
#    """Generator function that yields the
#    length of the strings in input_list."""
    # Yield the length of a string
#    for person in input_list:
#        yield len(person)
# Print the values generated by get_lengths()
#for value in get_lengths(lannister):
#    print(value)

#EXTRACTING A TIME COLUMN AND THEN EXTRACTING HOURS FROM IT - REAL DATA
# Extract the created_at column from df: tweet_time
#tweet_time = (df['created_at'])
# Extract the clock time: tweet_clock_time
#tweet_clock_time = [entry[11:19] for entry in tweet_time]
# Print the extracted times
#print(tweet_clock_time)

# Extract the created_at column from df: tweet_time
#tweet_time = df['created_at']
# Extract the clock time: tweet_clock_time
#tweet_clock_time = [entry[11:19] for entry in tweet_time if entry[17:19] == '19']
# Print the extracted times
#print(tweet_clock_time)
