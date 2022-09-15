import pandas as pd
data=pd.read_csv("C:/Users/kiit/Desktop/L&B data sets/50_Startups.csv")
print(data.isna().sum())
from sklearn.impute import SimpleImputer
SI=SimpleImputer()

SI_R__D=SI.fit(data[['R&D Spend']])
data['R&D Spend']=SI_R__D.transform(data[['R&D Spend']])


SI_Administration=SI.fit(data[['Administration']])
data['Administration']=SI_Administration.transform(data[['Administration']])


SI_Marketing_Spend=SI.fit(data[['Marketing Spend']])
data['Marketing Spend']=SI_Marketing_Spend.transform(data[['Marketing Spend']])


SI_State=SimpleImputer(strategy='most_frequent')
SI_State=SI_State.fit(data[['State']])
data['State']=SI_State.transform(data[['State']])
print(data.isna().sum())

print(data.corr()['Profit'])

from sklearn.preprocessing import LabelEncoder
LB=LabelEncoder()
data['State']=LB.fit_transform(data['State'])

X=data.iloc[:,[0,2]].values
Y=data.iloc[:,-1].values

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)


from sklearn.model_selection import train_test_split  
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.2,random_state=2)

print("PLOY,LR,KNN,SVR,DTR")

from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=2)
x_trans=poly.fit_transform(X_train)

from sklearn.linear_model import LinearRegression
Lr=LinearRegression()
Lr.fit(x_trans,Y_train)
x_trans_test=poly.fit_transform(X_test)
y1_pred=Lr.predict(x_trans_test)
from sklearn.metrics import r2_score
score1=r2_score(Y_test,y1_pred)
print(score1*100,'%')

from sklearn.neighbors import KNeighborsRegressor
KNN=KNeighborsRegressor(n_neighbors=3)
KNN.fit(X_train,Y_train)
y2_pred=KNN.predict(X_test)
from sklearn.metrics import r2_score
score2=r2_score(Y_test,y2_pred)
print(score2*100,'%')

from sklearn.svm import SVR
SVR=SVR(kernel='rbf',degree=2)
SVR.fit(X_train,Y_train)
y3_pred=SVR.predict(X_test)
from sklearn.metrics import r2_score
score3=r2_score(Y_test,y3_pred)
print(score3*100,'%')

from sklearn.tree import DecisionTreeRegressor
DTR=DecisionTreeRegressor(random_state=0,max_depth=4)
DTR.fit(X_train,Y_train)
y4_pred=DTR.predict(X_test)
from sklearn.metrics import r2_score
score4=r2_score(Y_test,y4_pred)
print(score4*100,'%')
