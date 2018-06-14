# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 21:13:40 2018

@author: mzhen
"""

# Python Crash Course 2
d = {'key1':'value','key2':123}

print(d['key1'])

print(d['key2'])

d = {'k1':[1,2,3]} 

print(d['k1'][0])

my_list = d['k1']

print(my_list[0])

# nested dictionary
d = {'k1':{'innerkey':[1,2,3]}}

# try to grab '2'

print(d['k1']['innerkey'][1])

my_list = [1,2,3]

# tuple
t = (1,2,3)

print(t)

# difference bw list and tuple
# re-asign value in my_list
my_list[0] = 'NEW'

print(my_list)

# cannot re-asign value in tuple, tuples are immutable, do not support 
# item re-asignment
# if you do not want user to change the item, use tuple

t[0] = 'NEW'
# expect error from the line above
# TypeError: 'tuple' object does not support item assignment

# Set is defined by only unique items.

set1 = {1,2,3}
print(set1)

set2 = {1,1,1,2,2,2,3,3,3}
print(set2)

# add item to a set
set2.add(6)
print(set2)

# cannot add an element that already exists in set2
set2.add(3)
print(set2)

# comparison operators e.g. =, >, <
1 > 2
1 < 2

# have to use two "=" signs, b/c one "=" means variable assignment
1 == 1
1 == 2

1 != 3

'high' == 'bye'

1 < 2 and 2 < 3
1 < 2 and 2 > 3

1 > 2 or 2 < 3
1 > 2 or 2 > 3

True or False
True and False

if 1<2:
    print('yep!')

if 1>2:
    print('no')
else:
    print('yep')

if 1 == 2:
    print('first')
else:
    print('last')    

if 1 == 2:
    print('first')
elif 3 == 3:
    print('yes')
else:
    print('last')    
   
# multiple elifs, only the first one get evaluated
if 1 == 2:
    print('first')
elif 3 == 3:
    print('yes')
elif 4 == 4:
    print('yes2')
else:
    print('last')   

seq = [1,2,3,4,5]

for item in seq:
    print(item)

# this will print 5 hello's    
for item in seq:
    print('hello')    
 
# this while loop below will run infinite numbers since i is always 1    
i = 1 

while i < 5:
    print('i is: {}'.format(i))    
#    
  
    
i = 1 

while i < 5:
    print('i is: {}'.format(i))       
    i = i+1
    
# range, defines the range between the number you want to start with and the 
# number you want to end with + 1     
range(0,5)

list(range(0,5))
# the list above will give [0, 1, 2, 3, 4]

for x in range(0,5):
    print(x)

# by default, a range starts from 0
range(10)    

x = [1,2,3,4]

#empty list
out = []

# for loop below will fill the empty list "out" with each value from x and square them

for num in x:
    out.append(num**2)    

print(out)

# translate the for loop above into a list comprehension
out1 = [num**2 for num in x]

print(out1)

# function 1

def my_func(param1):
    print(param1)
    
my_func('hello')

# function 2

def my_func1(name=''):
    print('Hello '+name)
       
my_func1('Jose')


def my_func2(name='Default Name'):
    print('Hello '+name)

my_func2()
my_func2('Miao')
my_func2(name='Miao')

# without brackets, python will list the the object
my_func

def square(num):
    return num**2

output = square(2)

print(output)

# documentation string can be added to a function
# nothing will change if you run the string
# the documentation will come out when you call out a signiture string with 
# control + i

def square(num):
    """
    THIS IS A DOCSTRING
    CAN GO MULTIPLE LINES
    THIS FUNCTION SQUARES A NUMBER
    """
    return num**2

# put cursor at the end of the function and press control + i
square


