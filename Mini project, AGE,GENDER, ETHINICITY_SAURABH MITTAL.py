#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')


# In[3]:


data = pd.read_csv("C:/Users/kiit/Downloads/FACE,AGE,ETHINICITY/age_gender.csv")
data.head()


# In[4]:


df = data.drop(['img_name','ethnicity'], axis=1)
df


# In[5]:


df.drop(df[df['age'] <15].index , inplace=True)


# In[6]:


indexAge = df[df['age'] >=25].index
df.drop(indexAge , inplace=True)


# In[7]:


df = df.drop('age', axis=1)
df


# In[8]:


def basic_eda(df):
    print("\n Shape: ")
    print(df.shape)
    print("\n\n --------- ")
    print("\n Number of null values: ")
    print(df.isnull().sum())
    print("\n\n --------- ")
    
    print("\n Value count of gender: ")
    print(df['gender'].value_counts())


# In[9]:


basic_eda(df)


# In[12]:


# Dividing target variables
y = df.drop("pixels", axis=1)
X = df.drop("gender", axis=1)
X.head()


# In[13]:


y.head()


# In[14]:


y.nunique()


# In[15]:


X = pd.Series(X['pixels'])
X = X.apply(lambda x:x.split(' '))
X = X.apply(lambda x:np.array(list(map(lambda z:np.int(z), x))))
X = np.array(X)
X = np.stack(np.array(X), axis=0)

# reshape data
X = X.reshape(-1, 48, 48, 1)
print("X shape: ", X.shape)


# In[16]:


X[0].shape


# In[17]:


plt.figure(figsize=(16,16))
for i,a in zip(np.random.randint(0, 3252, 25), range(1,26)):
    plt.subplot(5,5,a)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(np.squeeze(X[i]))
    plt.xlabel(str(y['gender'].iloc[i]))
plt.show()


# In[18]:


from sklearn.metrics import confusion_matrix
import itertools
import tensorflow as tf
import plotly.express as px
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical # convert to one-hot-encoding
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, InputLayer
from tensorflow.keras.optimizers import RMSprop,Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy


# In[19]:


gender_matrix = np.array(y['gender'])
gender = to_categorical(y["gender"], num_classes = 2)

print(gender)


# In[20]:


X=X/255


# In[21]:


datagen = ImageDataGenerator(
        featurewise_center = False,
    # set input mean to 0 over the dataset
       samplewise_center = False,
    # set each sample mean to 0 
       featurewise_std_normalization = False,
    # divide inputs by std of the dataset
       samplewise_std_normalization=False,  
    # divide each input by its std
       zca_whitening=False,
    # dimesion reduction
       rotation_range=5, 
    # randomly rotate images in the range 5 degrees
       zoom_range = 0.1,
    # Randomly zoom image 10%
       width_shift_range=0.1, 
    # randomly shift images horizontally 10%
       height_shift_range=0.1,  
    # randomly shift images vertically 10%
       horizontal_flip=False,  
    # randomly flip images
        vertical_flip=False  # randomly flip images
)

datagen.fit(X)


# In[22]:


from sklearn.model_selection import train_test_split
X_train_gender, X_test_gender, y_train_gender, y_test_gender = train_test_split(X, gender, test_size=0.2, random_state=42)

X_train_gender, X_val_gender, y_train_gender, y_val_gender = train_test_split(X_train_gender, y_train_gender, test_size=0.2, random_state=42)


# In[23]:


print(X_train_gender.shape)


# In[24]:


def my_model(num_classes, activation, loss):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding = "same", input_shape=(48,48,1)))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, kernel_size=(3,3),activation="relu",padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, kernel_size=(3,3),activation="relu",padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(256,activation="relu"))
    model.add(Dense(num_classes, activation=activation))
    
    model.compile(optimizer='Adam',
              loss= loss,
              metrics=['accuracy'])
    return model


# In[25]:


early_stopping = EarlyStopping(patience=10, 
                               min_delta=0.001,
                               restore_best_weights=True)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                           patience = 2,
                                           verbose=1,
                                           factor=0.5,
                                           min_lr = 0.00001)


# In[26]:


epochs = 4 
batch_size = 50
model_gender = my_model(2, "sigmoid", "binary_crossentropy")
history_gender = model_gender.fit(X_train_gender, y_train_gender, 
                                 batch_size = batch_size,
                                 epochs = epochs,
                                 validation_data = (X_val_gender, y_val_gender),
                                 steps_per_epoch = X_train_gender.shape[0] // batch_size, callbacks=[early_stopping,learning_rate_reduction])


# In[27]:


fig = px.line(
    history_gender.history, y=["loss", "val_loss"],
    labels = {'index':'epoch', 'value':'loss'},
    title = 'Training History')

fig.show()


# In[28]:


loss, acc = model_gender.evaluate(X_test_gender, y_test_gender, verbose=0)
print("Test Loss: {}".format(loss))
print("Test Accuracy: {}".format(acc))


# In[ ]:




