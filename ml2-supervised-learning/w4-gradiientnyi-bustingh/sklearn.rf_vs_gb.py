
# coding: utf-8

# # Sklearn, XGBoost

# ## sklearn.ensemble.RandomForestClassifier

# In[1]:

from sklearn import ensemble , cross_validation, learning_curve, metrics 

import numpy as np
import pandas as pd
import xgboost as xgb


# In[ ]:

get_ipython().magic('pylab inline')


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

# In[ ]:

bioresponce = pd.read_csv('bioresponse.csv', header=0, sep=',')


# In[ ]:

bioresponce.head()


# In[ ]:

bioresponce_target = bioresponce.Activity.values


# In[ ]:

bioresponce_data = bioresponce.iloc[:, 1:]


# ### Модель RandomForestClassifier

# #### Зависимость качества от количесвта деревьев

# In[ ]:

n_trees = [1] + range(10, 55, 5) 


# In[ ]:

get_ipython().run_cell_magic('time', '', "scoring = []\nfor n_tree in n_trees:\n    estimator = ensemble.RandomForestClassifier(n_estimators = n_tree, min_samples_split=5, random_state=1)\n    score = cross_validation.cross_val_score(estimator, bioresponce_data, bioresponce_target, \n                                             scoring = 'accuracy', cv = 3)    \n    scoring.append(score)\nscoring = np.asmatrix(scoring)")


# In[ ]:

scoring


# In[ ]:

pylab.plot(n_trees, scoring.mean(axis = 1), marker='.', label='RandomForest')
pylab.grid(True)
pylab.xlabel('n_trees')
pylab.ylabel('score')
pylab.title('Accuracy score')
pylab.legend(loc='lower right')


# #### Кривые обучения для деревьев большей глубины

# In[ ]:

get_ipython().run_cell_magic('time', '', "xgb_scoring = []\nfor n_tree in n_trees:\n    estimator = xgb.XGBClassifier(learning_rate=0.1, max_depth=5, n_estimators=n_tree, min_child_weight=3)\n    score = cross_validation.cross_val_score(estimator, bioresponce_data, bioresponce_target, \n                                             scoring = 'accuracy', cv = 3)    \n    xgb_scoring.append(score)\nxgb_scoring = np.asmatrix(xgb_scoring)")


# In[ ]:

xgb_scoring


# In[ ]:

pylab.plot(n_trees, scoring.mean(axis = 1), marker='.', label='RandomForest')
pylab.plot(n_trees, xgb_scoring.mean(axis = 1), marker='.', label='XGBoost')
pylab.grid(True)
pylab.xlabel('n_trees')
pylab.ylabel('score')
pylab.title('Accuracy score')
pylab.legend(loc='lower right')


# #### **Если Вас заинтересовал xgboost:**
# python api: http://xgboost.readthedocs.org/en/latest/python/python_api.html
# 
# установка: http://xgboost.readthedocs.org/en/latest/python/python_intro.html#install-xgboost
