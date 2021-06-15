#!/usr/bin/env python
# coding: utf-8

# In[1]:


##Importing the required modules

import warnings
warnings.filterwarnings('ignore')

##Determining whether the warnings to be issued in the program is controlled by filterwarnings()

import numpy as np                                                       #for numerical operation in python
import pandas as pd                                                      #for DataFramework 
import math                                                              #for mathematics operation
from sklearn.metrics import accuracy_score                               
from sklearn.model_selection import train_test_split
import keras                                                             #for Neural Networking
import matplotlib.pyplot as plt                                          #for plotting graphs and models
import seaborn as sns                                                    #for Statistical Data visualization


# In[2]:


##Reading the data from a .csv file

data=pd.read_csv(r'C:\Users\admin\Downloads\Mortality+Prediction+in+ICU\Mortality Prediction in ICU/train.csv')
labels=pd.read_csv(r'C:\Users\admin\Downloads\Mortality+Prediction+in+ICU\Mortality Prediction in ICU/labels.csv')


# In[3]:


#display of first five rows of the data

print(data.head())
print(labels.head())


# In[4]:


#Collecting the columns/labels in data

print(data.columns)
print(labels.columns)


# In[5]:


#Shape of the data

print(data.shape)
print(labels.shape)


# In[6]:


#Description of the data

print(data.describe())


# In[7]:


#General information of the data

data.info()


# In[8]:


#Counting number of deaths(ones) in labels

labels['In-hospital_death'].value_counts()


# In[9]:


#One hot encoding to make the model prediction better

temp=[]
for i in labels['In-hospital_death']:
    if i==0:
        temp.append([1,0])
    else:
        temp.append([0,1])
temp=np.array(temp)                    #After one-hot-encoding the death will be denoted as '1' and not-death iwill be denoted as '0'


# In[10]:


print(temp.shape)


# In[11]:


#concatenating data and labels

con_data=pd.concat([data,labels],axis=1)         #axis=1 represents concatinating labels to data columnwise
con_data.shape


# In[12]:


#Using Correlation heatmap to find the important 

corr_map=con_data[con_data.columns].corr()
obj=np.array(corr_map)
obj[np.tril_indices_from(obj)]=False

fig,ax=plt.subplots()
fig.set_size_inches(25,18)
sns.heatmap(corr_map,mask=obj,vmax=.7,square=True,annot=True,linewidths=0.001)


# In[13]:


##After plotting the heatmap we can find the least correlated features and can remove them for better performance


# In[14]:


new_data=con_data.drop(['In-hospital_death'],axis=1)            #Dropping the labels


# In[15]:


data=new_data.drop(['Gender','Cholesterol','HCT','ICUType','Height'],axis=1)      #Dropping the least correlated features


# In[18]:


#Scaling the data

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
data=scaler.fit_transform(data)


# In[20]:


##Just assigning the data and labels as X and y for our convinience

X=data
y=temp


# In[28]:


##Splitting the Dataset

x_train,x_test,y_train,y_test=train_test_split(X,y)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)                                      #Shape of training and testing datasets


# In[33]:


##Importing necessory files for NN implementation

from keras.models import Sequential                         #Sequential allows the user to create layer-by-layer model. In this each layer has exactly one input and one output
from keras.layers import Dense,Dropout,BatchNormalization   #Dense is the basic layer of NN, it feeds all the outputs from previous layer to all the neurons
from keras.utils import np_utils                            #BatchNormalization applies the transformation that maintains a mean output to 0
from keras.optimizers import RMSprop,Adam


# In[41]:


##Building ANN model

model=Sequential()
model.add(Dense(64,input_dim=x_train.shape[1],activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(196,activation='relu'))
model.add(Dense(196,activation='relu'))

model.add(BatchNormalization())

model.add(Dense(256,activation='relu'))
model.add(Dense(2,activation='sigmoid'))

model.compile(optimizer=Adam(lr=0.0005),loss='binary_crossentropy',metrics=['accuracy'])
print(model.summary())


# In[42]:


##Fitting the model

predict_death=model.fit(x_train,y_train,epochs=15,batch_size=128,validation_data=(x_test,y_test))


# In[48]:


##Evaluating the performance of the model

print(predict_death.history.keys())         #History is a default callback that records training metrics for each epochs includes loss and accuracy


# In[49]:


plt.plot(predict_death.history['accuracy'])


# In[51]:


plt.plot(predict_death.history['val_accuracy'])


# In[52]:


plt.plot(predict_death.history['loss'])


# In[53]:


plt.plot(predict_death.history['val_loss'])


# In[56]:


## Confusion matrix

from sklearn.metrics import confusion_matrix
pred=model.predict(x_test)
pred=np.argmax(pred,axis=1)
y_true=np.argmax(y_test,axis=1)


# In[58]:


cnf_matrix=confusion_matrix(y_true,pred)
print(cnf_matrix)


# In[59]:


##Accuracy score of the model

print(accuracy_score(y_true,pred))


# In[ ]:




