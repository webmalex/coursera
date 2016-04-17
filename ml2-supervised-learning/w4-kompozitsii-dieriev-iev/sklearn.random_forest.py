
# coding: utf-8

# # Sklearn

# ## sklearn.ensemble.RandomForestClassifier

# документация:  http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

# In[1]:

get_ipython().magic('pylab inline')


# In[2]:

from sklearn import ensemble, cross_validation, learning_curve, metrics 

import numpy as np
import pandas as pd


# ### Данные

# Задача на kaggle: https://www.kaggle.com/c/bioresponse
# 
# Данные: https://www.kaggle.com/c/bioresponse/data
# 
# По данным характеристикам молекулы требуется определить, будет ли дан биологический ответ (biological response).
# 
# Признаки нормализаваны.
# 
# Для демонстрации используется обучающая выборка из исходных данных train.csv, файл с данными прилагается.

# In[3]:

bioresponce = pd.read_csv('bioresponse.csv', header=0, sep=',')


# In[4]:

bioresponce.head()


# In[5]:

bioresponce.shape


# In[6]:

bioresponce.columns


# In[7]:

bioresponce_target = bioresponce.Activity.values


# In[8]:

print 'bioresponse = 1: {:.2f}\nbioresponse = 0: {:.2f}'.format(sum(bioresponce_target)/float(len(bioresponce_target)), 
                1.0 - sum(bioresponce_target)/float(len(bioresponce_target)))


# In[9]:

bioresponce_data = bioresponce.iloc[:, 1:]


# ### Модель RandomForestClassifier

# #### Кривые обучения для деревьев небольшой глубиной 

# In[10]:

rf_classifier_low_depth = ensemble.RandomForestClassifier(n_estimators = 50, max_depth = 2, random_state = 1)


# In[11]:

train_sizes, train_scores, test_scores = learning_curve.learning_curve(rf_classifier_low_depth, bioresponce_data, bioresponce_target, 
                                                                       train_sizes=np.arange(0.1,1., 0.2), 
                                                                       cv=3, scoring='accuracy')


# In[12]:

print train_sizes
print train_scores.mean(axis = 1)
print test_scores.mean(axis = 1)


# In[13]:

pylab.grid(True)
pylab.plot(train_sizes, train_scores.mean(axis = 1), 'g-', marker='o', label='train')
pylab.plot(train_sizes, test_scores.mean(axis = 1), 'r-', marker='o', label='test')
pylab.ylim((0.0, 1.05))
pylab.legend(loc='lower right')


# #### Кривые обучения для деревьев большей глубины

# In[14]:

rf_classifier = ensemble.RandomForestClassifier(n_estimators = 50, max_depth = 10, random_state = 1)


# In[15]:

train_sizes, train_scores, test_scores = learning_curve.learning_curve(rf_classifier, bioresponce_data, bioresponce_target, 
                                                                       train_sizes=np.arange(0.1,1, 0.2), 
                                                                       cv=3, scoring='accuracy')


# In[16]:

pylab.grid(True)
pylab.plot(train_sizes, train_scores.mean(axis = 1), 'g-', marker='o', label='train')
pylab.plot(train_sizes, test_scores.mean(axis = 1), 'r-', marker='o', label='test')
pylab.ylim((0.0, 1.05))
pylab.legend(loc='lower right')


# In[ ]:



