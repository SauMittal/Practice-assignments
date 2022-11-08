# Removing of character from the string

word = "saurabh"
print(word.replace("a",""))


# method 2

#using Translate

x= "saurabh"
a="a"
b="b"

word = x.maketrans(a,b)
print(x.translate(word))


# Possible combination in a string

m = "123"
n = [m[i: j]

for i in range (len(m))
     for j in range(i+1, len(m) +1)]
print(n)
print(len(n))

# using permutation

def perm(string, i=0):
    if i==len(string):
        print("".join(string))

    for j in range(i, len(string)):
        X=[a for a in string]

        X[i],X[j] = X[j],X[i]
        perm(X,i+1)
print(perm("RAM"))
