# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 12:15:15 2022

@author: kiit
"""
import pandas as pd
data=pd.read_csv("C:/Users/kiit/Downloads/IRIS.csv")


print(data.head())
print(data.isna())
X=data.iloc[:,0:4].values # Here we are not converting it in numpy array
Y=data.iloc[:,-1].values


# Reshape of rows 
Y=Y.reshape(Y.shape[0],1)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.3,random_state=23)

# check for KNeighbourClassifier
from sklearn.neighbors import KNeighborsClassifier
value=[]
for i in range(1,10):
    
    model=KNeighborsClassifier(n_neighbors=i)
    model.fit(X_train,Y_train)
    Y_pred=model.predict(X_test)
    from sklearn.metrics import confusion_matrix
    cm=confusion_matrix(Y_test, Y_pred)
    
    from sklearn.metrics import accuracy_score
    acc=accuracy_score(Y_pred, Y_test)    
    value.append(acc)
X1=[1,2,3,4,5,6,7,8,9]
import matplotlib.pyplot as plt
plt.plot(X1,value,color='b',label='accuracy score')
plt.legend()
plt.show()

m={'Iris-setosa':'red','Iris-versicolor':'blue','Iris-virginica':'green'}
plt.scatter(x=data['sepal_length'],y=data['sepal_width'],color=data['species'].map(m))
plt.show()
            

'''from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
CT=ColumnTransformer([('OHE',OneHotEncoder(drop='first'),[-1])],remainder='passthrough')
data=CT.fit_transform(data)
x=data[:,4:6]
y=data[:,0:2]
import matplotlib.pyplot as plt
plt.scatter(x,y)
plt.show()'''



'''a=data['sepal_length']
b=a.mean()
print(b)
sepal_mean_value=a[a>b]
print(sepal_mean_value)

A=data['sepal_width']
B=A.mean()
print(B)
sepal_width_mean=A[A>B]
print(sepal_width_mean)

s=data['sepal_length']
S=s.median()
print(S)
z=a[s>S]
print(z)'''