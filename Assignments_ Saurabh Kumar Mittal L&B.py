#Noraml calculator
'''crete a calculator'''
'''x=int(input('enter the 1st no.'))
y=int(input('enter the 2nd no.'))
c=input('enter the sign')

if c=='+':
    print(x+y)
if c=='-':
    print(x-y)
if c=='/':
    print(x/y)
if c=='*':
    print(x*y)'''
'''a=input("select operations from +,-,/,*:")
number1=int(input('enter the 1st no.'))
number2=int(input('enter the second no.'))

if a=='+':
    print(number1+number2) 
elif a=='-':
    print(number1-number2)
elif a=='/':
    print(number1/number2)
elif a=='*':
    print(number1*number2)'''

# Tuple taking 10 user input
'''create a tuple and take 10 user input
b=[]
for i in range(0,10):
    a=input("enter the value")
    b.append(a)
b=tuple(b)
print(b)'''

# Leap year or not
'''create a program to find a year is leap year or not
a=int(input("enter the year"))
if(a%4==0):
    print("a leap year")
else:
        print("not a leap year")'''

# Palindrome or not
'''create a program to find out the no. is palindrom   
num=int(input("enter the number"))
a=num
reverse=0
while(num>0):
    b=num%10
    reverse=reverse*10+b
    num=num//10
if(a==reverse):
    print("palindrome")
else:
    print("not palindrome")'''


# Palindrome checking words
'''a=input('enter the word')
rev=reversed(a)
if(list(a)==list(rev)):
    print('palindrome')
else:
    print('not palindrome')'''

'''a=input("enter the word")
b=""
for i in a:
    b=i+b
if (a==b):
    print("palindrome")
else:
    print("not palindrome")'''

# Scientific calculator
'''import math
a=int(input("enter the no."))
b=int(input("enter the no."))
c=input("enter the sign")
if(c=='+'):
    print(a+b)
if(c=='-'):
    print(a-b)
if(c=='*'):
    print(a*b)
if(c=='/'):
    print(a/b)
x=int(input())
if(c=='sin'):
   print(math.sin(x))
x=int(input())
if(c=='cos'):
   print(math.cos(x))
x=int(input())
if(c=='tan'):
   print(math.tan(x))'''
''''import math
x=int(input('enter the sin value'))
p=(math.sin(math.radians(x)))
print(p)
x=int(input('enter the cosec value'))
p=(math.sin(math.radians(x)))
d=1/p
print(d)'''

# Bitwise operator
'''bit wise operaror
while(1):
    def bitwise(a,b):
        c=input("enter bitwise and / or")
        if(c=='&'):
            print("the AND of a and b is",a&b)
        elif(c=='|'):
            print("the OR of a and b is", a|b)

    bitwise(1,2)'''


#Fibonacci series 
'''fibonacci series
def fib(a):
    b=0
    c=1
    if a==1:
        print(a)
    else:
        print(a)
        print(b)
        for i in range(2,a):
            z=b+c
            b=c
            c=z
            print(c)'''

'''fubonacci series
while(1):
    a=int(input("enter the no. of series"))
    x=0
    y=1
    if a<=0:
        print("the number is",x)
    else:
        print(x)
        print(y)
        for i in range(2,a):
            b=x+y
            x=y
            y=b
            print(y)'''
# Prime no. or not
'''Prime or not
while(1):
    num=int(input('enter the no.'))

    for i in range(2,num):
        if(num%i==0):
            print('not prime')
            break
    else:
        print('a prime')'''

'''scientific calculator
while(1):
    import math
    print('1.basic')
    print('2.scientific')
    h=input('enter the calc')
    if(h=='1'):
        a=input("enter the type")
        x=int(input('number 1:'))
        y=int(input('number 2:'))

        if(a=='add'):
            print(x+y)

        if (a=='sub'):
            print(x-y)

        if (a=='multiply'):
            print(x*y)

        if (a=='divide'):
            print(x/y)

    if(h=='2'):
        a=input("enter the values")
        b=int(input("enter the values in degree"))
        if(a=='sin'):
            print(math.sin(math.radians(b)))          
        if(a=='cos'):
            print(math.cos(math.radians(b)))
        if (a=='tan'):
            print(math.tan(math.radians(b)))
        if(a=='cosec'):
            x=math.sin(math.radians(b))
            c=1/x
            print(c)
        if (a=='sec'):
            x=math.cos(math.radians(b))
            c=1/x
            print(c)
        if (a=='cot'):
            x=math.tan(math.radians(b))
            c=1/x
            print(c)
        if (a=='factorial'):
            print(math.factorial(b))
        if (a=='log'):
            print(math.log(b))'''


'''lst={}
for i in range(3):
    a=input("enter key")
    b=input("enter values")
    lst[a]=b
print(lst)'''
# Even and odd no.
'''even or odd number
while(1):
    a=int(input("enter the number"))
    if (a%2==0):
        print("even")
    else:
        print("odd")'''

'''fibonacci series

a=int(input("enter the no."))
b=int(input("enter the no."))
c=int(input("enter the series"))
w=(c-2)
print(a)
print(b)
for i in range(w):
    x=a+b
    a=b
    b=x
    print(b)'''



     

   


