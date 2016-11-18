
# coding: utf-8

# In[1]:

#input: h5, features: h5.analysis
#output: 100*4973, like

#training 2350 sounds
#methodoloty:
#1.Similarity: recommendation system
#2.regression: f(sounds) -> lyrics, KNN, RF
#3.ANN


# In[154]:

get_ipython().magic(u'matplotlib inline')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import sklearn

from pandas.io.data import DataReader
from sklearn.linear_model import LogisticRegression
from sklearn.lda import LDA
from sklearn.qda import QDA
from keras.models import Sequential
from keras.layers import LSTM, Dense
import random
from sklearn import datasets, linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from itertools import product
from sklearn.ensemble import VotingClassifier
from sklearn import preprocessing
from sklearn import cross_validation
from scipy.stats import spearmanr 
import xgboost as xgb
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score


# In[155]:

import h5py
import pandas as pd
import os
import numpy as np


# In[232]:

lyr = pd.read_csv('Project4_data/lyr.csv', index_col=0)


# In[233]:

#usefulcolumns = [0,3,4]
#usefulcolumns.extend(range(30,5001))


# In[234]:

#lyr = lyr.iloc[:,:]


# In[235]:

lyr.shape #columns: 2973 words


# In[236]:

lyr.shape[0]


# In[53]:

#import data
#array
old = [2,6,7,10,12,15]
X = []
labels = []
for root, dirs, filenames in os.walk('Project4_data/data'):
    for filename in filenames:
        if filename != '.DS_Store':
            f = h5py.File(root +'/'+ filename, 'r')[u'analysis'].values()
            n = len(f)
            temp = []
            for index in old:
                temp.append(f[index])
                #labels.append(filename[index])
            X.append(temp)
#problem
X = np.array(X)
#X.reshape(100,5789)
X.shape


# In[39]:

#truncate data


# In[52]:

dim = []
for j in range(X.shape[1]):
    dim = []
    for i in range(X.shape[0]):
        dim.append(X[i][j].shape[0])
    print 'the %s dimension has lenth'%j, min(dim)


# In[54]:

X.shape


# In[237]:

X_new = []
y_new = []
y = np.array([x[1:] for x in np.array(lyr)])
for i in range(X.shape[0]):
    temp = []
    for j in range(X.shape[1]):
        try:
            temp.extend(X[i][j].reshape(-1,)[:100])
        except:
            pass
    if len(temp) == 600:
        X_new.append(temp)
    
        temp = y[i]
        total = sum(temp) + 0.0
        for j in range(len(temp)):
            temp[j] = temp[j] / total
        y_new.append(temp)
    
X_new = np.array(X_new)
y_new = np.array(y_new)


# In[238]:

print X_new.shape
print y_new.shape


# In[239]:

words = (list(lyr.columns)[1:]) #column names of Y


# In[240]:

from sklearn import model_selection


# In[241]:

#split training and testing data
X_train, X_test, y_train, y_test = model_selection.train_test_split(X_new, y_new, test_size=0.33, random_state=42)


# In[242]:

from sklearn import neighbors
knn = neighbors.KNeighborsRegressor(7)
knn = knn.fit(X_new, y_new)


# In[243]:

#import data
#array
new = [2,6,7,10,12,14]
X1 = []
labels = []
for root, dirs, filenames in os.walk('Project4_data/TestSongFile100/'):
    for filename in filenames:
        if filename != '.DS_Store':
            f = h5py.File(root +'/'+ filename, 'r')[u'analysis'].values()
            n = len(f)
            temp = []
            for index in new:
                temp.append(f[index][:])
                #labels.append(filename[index])
            X1.append(temp)
#problem
X1 = np.array(X1)
#X.reshape(100,5789)
X1.shape


# In[244]:

X_new1 = []
for i in range(X1.shape[0]):
    temp = []
    for j in range(X1.shape[1]):
        try:
            temp.extend(X1[i][j].reshape(-1,)[:100])
        except:
            pass
    X_new1.append(temp)
    
    
X_new1 = np.array(X_new1)


# In[245]:

y_pred_new = knn.predict(X_new1)


# In[259]:

for i in range(100):
    y_pred_new[i][0] = 0
    y_pred_new[i] = y_pred_new[i].argsort()


# In[262]:

result_new = pd.DataFrame(y_pred_new)
result_new.columns = list(lyr.columns[1:])


# In[264]:

result_new.to_csv('submission.csv')


# In[ ]:



