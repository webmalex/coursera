
# coding: utf-8

# # Sklearn

# ## sklearn.liner_model

# **linear_model:**
# * RidgeClassifier
# * SGDClassifier
# * SGDRegressor
# * LinearRegression
# * LogisticRegression
# * Lasso
# * etc

# документация: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model
# 
# примеры: http://scikit-learn.org/stable/modules/linear_model.html#linear-model

# In[1]:

from matplotlib.colors import ListedColormap
from sklearn import cross_validation, datasets, linear_model, metrics

import numpy as np


# In[2]:

get_ipython().magic('pylab inline')


# ### Генерация данных

# In[3]:

blobs = datasets.make_blobs(centers = 2, cluster_std = 5.5, random_state=1)


# In[4]:

colors = ListedColormap(['red', 'blue'])

pylab.figure(figsize(8, 8))
pylab.scatter(map(lambda x: x[0], blobs[0]), map(lambda x: x[1], blobs[0]), c = blobs[1], cmap = colors)


# In[5]:

train_data, test_data, train_labels, test_labels = cross_validation.train_test_split(blobs[0], blobs[1], 
                                                                                    test_size = 0.3,
                                                                                    random_state = 1)


# ### Линейная классификация

# #### RidgeClassifier

# In[6]:

#создание объекта - классификатора
ridge_classifier = linear_model.RidgeClassifier(random_state = 1)


# In[7]:

#обучение классификатора
ridge_classifier.fit(train_data, train_labels)


# In[8]:

#применение обученного классификатора
ridge_predictions = ridge_classifier.predict(test_data)


# In[9]:

print test_labels


# In[10]:

print ridge_predictions


# In[11]:

#оценка качества классификации
metrics.accuracy_score(test_labels, ridge_predictions)


# In[12]:

ridge_classifier.coef_


# In[13]:

ridge_classifier.intercept_ 


# #### LogisticRegression

# In[14]:

log_regressor = linear_model.LogisticRegression(random_state = 1)


# In[15]:

log_regressor.fit(train_data, train_labels)


# In[16]:

lr_predictions = log_regressor.predict(test_data)


# In[17]:

lr_proba_predictions = log_regressor.predict_proba(test_data)


# In[18]:

print test_labels


# In[19]:

print lr_predictions


# In[20]:

print lr_proba_predictions


# In[21]:

print metrics.accuracy_score(test_labels, lr_predictions)


# In[22]:

print metrics.accuracy_score(test_labels, ridge_predictions)


# ### Оценка качества по cross-validation

# #### cross_val_score

# In[23]:

ridge_scoring = cross_validation.cross_val_score(ridge_classifier, blobs[0], blobs[1], scoring = 'accuracy', cv = 10)


# In[24]:

lr_scoring = cross_validation.cross_val_score(log_regressor, blobs[0], blobs[1], scoring = 'accuracy', cv = 10)


# In[25]:

lr_scoring


# In[26]:

print 'Ridge mean:{}, max:{}, min:{}, std:{}'.format(ridge_scoring.mean(), ridge_scoring.max(), 
                                                     ridge_scoring.min(), ridge_scoring.std())


# In[27]:

print 'Log mean:{}, max:{}, min:{}, std:{}'.format(lr_scoring.mean(), lr_scoring.max(), 
                                                   lr_scoring.min(), lr_scoring.std())


# #### cross_val_score с заданными scorer и cv_strategy

# In[28]:

scorer = metrics.make_scorer(metrics.accuracy_score)


# In[29]:

cv_strategy = cross_validation.StratifiedShuffleSplit(blobs[1], n_iter = 20 , test_size = 0.3, random_state = 2)


# In[30]:

ridge_scoring = cross_validation.cross_val_score(ridge_classifier, blobs[0], blobs[1], scoring = scorer, cv = cv_strategy)


# In[31]:

lr_scoring = cross_validation.cross_val_score(log_regressor, blobs[0], blobs[1], scoring = scorer, cv = cv_strategy)


# In[32]:

print 'Ridge mean:{}, max:{}, min:{}, std:{}'.format(ridge_scoring.mean(), ridge_scoring.max(), 
                                                     ridge_scoring.min(), ridge_scoring.std())


# In[33]:

print 'Log mean:{}, max:{}, min:{}, std:{}'.format(lr_scoring.mean(), lr_scoring.max(), 
                                                   lr_scoring.min(), lr_scoring.std())

