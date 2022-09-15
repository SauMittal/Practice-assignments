#Saurabh Kumar Mittal

import numpy as np
import pandas as pd

data=pd.read_csv("C:/Users/kiit/Desktop/L&B data sets/IMDB-Movie-Data.csv")
#print(data)
print(data.info())
#print(data.head())
#print(data.tail())
#print(data.columns)
#print(data['Votes'].unique())
print(data.isna().sum())
print(data.describe())
print(data['Votes'].mean())

print(data[data['Votes']>169809.255].shape)
#print(data['Votes']>169809.255)

print(data['Rating'].mean())
print(data[data['Rating']>8].shape)
print(data['Runtime (Minutes)'].mean())
print(data[data['Runtime (Minutes)']<113.172].shape)


#group_ver=data.groupby("Votes")
#print(group_ver.mean())

