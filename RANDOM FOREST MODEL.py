import pandas as pd
data=pd.read_csv("C:/Users/kiit/OneDrive - Brandscapes Consultancy Pvt. Ltd/Desktop/L&B data sets/Social_Network_Ads.csv")
print(data)
print(data.head())
print(data.isna())

X=data.iloc[:,1:4].values
Y=data.iloc[:,-1].values
Y=Y.reshape(Y.shape[0],1)

from sklearn.preprocessing import OneHotEncoder as o
from sklearn.compose import ColumnTransformer as c
CT=c([('OHE',o(drop='first'),[0])],remainder='passthrough')
X=CT.fit_transform(X)

from sklearn.preprocessing import StandardScaler as s
sc=s()
X=sc.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.2,random_state=23)

from sklearn.ensemble import RandomForestClassifier as r
value=[]
for i in range(1,10):
    model=r(n_estimators=i)
    model.fit(X_train,Y_train)
    Y_pred=model.predict(X_test)
    
    from sklearn.metrics import confusion_matrix 
    cm=confusion_matrix(Y_test,Y_pred)
    print(cm)
    
    from sklearn.metrics import accuracy_score
    acc=accuracy_score(Y_test,Y_pred)
    print(acc)
    value.append(acc)
X1=[1,2,3,4,5,6,7,8,9]
import matplotlib.pyplot as plt
plt.plot(X1,value,color='b',label='accuracy score')
plt.legend()
plt.show()



