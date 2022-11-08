# -*- coding: utf-8 -*-
"""Review_RNN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1yjOzPqWkapHOiO6z-i4KXdZEeppkpduV
"""

import numpy as np
import pandas as pd

df = pd.read_csv("/content/drive/MyDrive/Saurabh Kumar Mittal_Assignment /Saurabh Mittal Assignment + Projects of L&B/RNN PROJECT/Restaurant_Reviews.csv")
df.head()

from google.colab import drive
drive.mount('/content/drive')

df.isna().sum()

X=df["Review"]
Y=df["Liked"]

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.2,random_state=23)

print("X_train shape:",X_train.shape)
print("Y_train shape:",Y_train.shape)
print("X_test shape:",X_test.shape)
print("Y_test shape:",Y_test.shape)

print(X_train[7])
print(Y_train[7])

from keras.preprocessing.text import Tokenizer
tokenizer= Tokenizer(num_words=1000,lower=True)
tokenizer.fit_on_texts(X_train)
X_train=tokenizer.texts_to_sequences(X_train)
X_test=tokenizer.texts_to_sequences(X_test)
vocab_size=len(tokenizer.word_index)+1

print(X_train[7])
print(Y_train[7])

from keras_preprocessing.sequence import pad_sequences
X_train=pad_sequences(X_train, maxlen=100,padding="post")
X_test=pad_sequences(X_test, maxlen=100,padding="post")
print(X_train[7:])

from keras.utils import to_categorical
num_classes=2
Y_train=to_categorical(Y_train,num_classes)
Y_test=to_categorical(Y_test,num_classes)
print(Y_train.shape)
print(Y_train[7])

print(Y_train)

from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,SimpleRNN
from keras import optimizers

X_train=np.array(X_train).reshape((X_train.shape[0],X_train.shape[1],1))
X_test=np.array(X_test).reshape((X_test.shape[0],X_test.shape[1],1))
print(X_train.shape)
print(X_test.shape)

model=Sequential()
model.add(SimpleRNN(50,input_shape=(100,1),return_sequences=False))
model.add(Dense(num_classes))
model.add(Activation("softmax"))
model.summary()
adam= optimizers.Adam(learning_rate=0.001)
model.compile(loss="categorical_crossentropy",optimizer=adam,metrics=["accuracy"])

history=model.fit(X_train,Y_train,epochs=50,batch_size=64,validation_data=(X_test,Y_test))

import matplotlib.pyplot as plt
plt.figure(figsize=(16,5))
epochs=range(1,len(history.history["accuracy"])+1)
plt.plot(epochs,history.history["loss"],"b",label="Traning Loss",color="red")
plt.plot(epochs,history.history["val_loss"],"b",label="Validation Loss")
plt.legend()
plt.show()

plt.figure(figsize=(16,5))
epochs=range(1,len(history.history["accuracy"])+1)
plt.plot(epochs,history.history["accuracy"],"b",label="Traning accuracy",color="red")
plt.plot(epochs,history.history["val_accuracy"],"b",label="Validation accuracy")
plt.legend()
plt.show()

y_pred=model.predict(X_test)
Y_test_=np.argmax(Y_test,axis=1)

print(Y_test_)
print(y_pred)

Validation_sentence=["I dont like food."]
Validation_sentence=tokenizer.texts_to_sequences(Validation_sentence)
Validation_sentence=np.array(Validation_sentence)
Validation_sentence=pad_sequences(Validation_sentence,maxlen=100,padding="post")
Validation_sentence=Validation_sentence.reshape((Validation_sentence.shape[0],Validation_sentence.shape[1],1))
prediction=model.predict(np.array(Validation_sentence))
VS=np.argmax(prediction,axis=1)
print("Output of the model is",VS)
if VS==0:
  print("It is negative feedback")
elif VS==1:
  print("It is positive feedback")

Validation_sentence=[" It was slow service."]
Validation_sentence=tokenizer.texts_to_sequences(Validation_sentence)
Validation_sentence=np.array(Validation_sentence)
Validation_sentence=pad_sequences(Validation_sentence,maxlen=100,padding="post")
Validation_sentence=Validation_sentence.reshape((Validation_sentence.shape[0],Validation_sentence.shape[1],1))
prediction=model.predict(np.array(Validation_sentence))
VS=np.argmax(prediction,axis=1)
print("Output of the model is",VS)
if VS==1:
  print("It is Positive feedback")
else:
  print("It is negative feedback")

Validation_sentence=[" Wow... Loved this place."]
Validation_sentence=tokenizer.texts_to_sequences(Validation_sentence)
Validation_sentence=np.array(Validation_sentence)
Validation_sentence=pad_sequences(Validation_sentence,maxlen=100,padding="post")
Validation_sentence=Validation_sentence.reshape((Validation_sentence.shape[0],Validation_sentence.shape[1],1))
prediction=model.predict(np.array(Validation_sentence))
VS=np.argmax(prediction,axis=1)
print("Output of the model is",VS)
if VS==1:
  print("It is Positive feedback")
else:
  print("It is negative feedback")

