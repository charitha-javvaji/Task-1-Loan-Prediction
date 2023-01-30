#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Project Name:
    
                             # Loan Status Prediction with Python


# In[ ]:


# Problem Statement:
# Consider there is a finance company that gives loan for people,before approving the loan this company analysis various credentials of the persons?
                # (using MACHINE LEARNING)


# In[ ]:


# Work Flow:
# 1) we need some data
# 2) Data preprocessing
# 3) Split the data(train,test)
# 4) Support vector machine model(loan approved,rejected)


# In[ ]:


# Importing the dependencies 


# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# In[ ]:


# Data collection and processing
    # Loading the dataset to pandas DataFrame


# In[4]:


loan_dataset=pd.read_csv("C:\\Users\\91939\\Downloads\\archive.zip")


# In[5]:


type(loan_dataset)


# In[6]:


# Printing first 5 rows of the data sets
    
loan_dataset.head()    


# In[8]:


# Number of rows and columns
loan_dataset.shape


# In[9]:


# To Print the description of data
loan_dataset.describe()    


# In[12]:


# To check the info of the data
loan_dataset.info()


# In[13]:


# Number of missing values in each column
loan_dataset.isnull().sum()    


# In[14]:


# Droping the missing values
loan_dataset=loan_dataset.dropna()    


# In[15]:


loan_dataset.isnull().sum()   


# In[16]:


# Lable encoding
loan_dataset.replace({'Loan_Status':{'N':0,'Y':1}},inplace=True)    


# In[17]:


loan_dataset.head()  


# In[18]:


# Dependent column values
loan_dataset['Dependents'].value_counts()    


# In[19]:


# Replacing the value of 3+ to 4
loan_dataset=loan_dataset.replace(to_replace='3+',value=4)    


# In[20]:


loan_dataset['Dependents'].value_counts()  


# In[ ]:


# Data visualizations


# In[22]:


# Education and loan status
sns.countplot(x='Education',hue='Loan_Status',data=loan_dataset)    


# In[23]:


# Marital status and loan status
sns.countplot(x='Married',hue='Loan_Status',data=loan_dataset)     


# In[ ]:


# To Make the Pairplot of our Entire Dataset in once


# In[38]:


sns.pairplot(loan_dataset)


# In[24]:


# Convert categorical to numerical columns
loan_dataset.replace({'Married':{'No':0,'Y':1},'Gender':{'Male':1,'Female':0},'Self_Employed':{'No':0,'Y':1},
              'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2},'Education':{'Graduate':1,'Not Graduate':0,}},inplace=True)    


# In[25]:


loan_dataset.head()  


# In[26]:


# Seperating the data and lables
    
x=loan_dataset.drop(columns=['Loan_ID','Loan_Status'],axis=1)
y=loan_dataset['Loan_Status']


# In[27]:


print(x)
print(y)


# In[29]:


# Spliting Train and Test 
    
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.1,stratify=y,random_state=2)    


# In[31]:


print(x.shape,x_train.shape,x_test.shape)


# In[ ]:


# Training the model:
# Support vector machne model


# In[34]:


classifier=svm.SVC(kernel='linear')


# In[35]:


# Training the support vector machine model


# In[36]:


classifier.fit(x_train,y_train)


# In[ ]:


# Overall data set


# In[37]:


loan_dataset


# In[ ]:


#--------------THE-END--------------#                 
         
    

