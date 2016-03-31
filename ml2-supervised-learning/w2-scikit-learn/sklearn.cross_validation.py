
# coding: utf-8

# # Sklearn

# ## sklearn.cross_validation

# документация: http://scikit-learn.org/stable/modules/cross_validation.html

# In[1]:

from sklearn import cross_validation, datasets

import numpy as np


# ### Разовое разбиение данных на обучение и тест с помощью train_test_split

# In[2]:

iris = datasets.load_iris()


# In[3]:

train_data, test_data, train_labels, test_labels = cross_validation.train_test_split(iris.data, iris.target, 
                                                                                     test_size = 0.3)


# In[4]:

#убедимся, что тестовая выборка действительно составляет 0.3 от всех данных
float(len(test_labels))/len(iris.data)


# In[5]:

print 'Размер обучающей выборки: {} объектов \nРазмер тестовой выборки: {} объектов'.format(len(train_data),
                                                                                            len(test_data))


# In[6]:

print 'Обучающая выборка:\n', train_data[:5]
print '\n'
print 'Тестовая выборка:\n', test_data[:5]


# In[7]:

print 'Метки классов на обучающей выборке:\n', train_labels
print '\n'
print 'Метки классов на тестовой выборке:\n', test_labels


# ### Стратегии проведения кросс-валидации

# #### KFold

# In[8]:

for train_indices, test_indices in cross_validation.KFold(10, n_folds = 5):
    print train_indices, test_indices


# In[9]:

for train_indices, test_indices in cross_validation.KFold(10, n_folds = 2, shuffle = True):
    print train_indices, test_indices


# In[10]:

for train_indices, test_indices in cross_validation.KFold(10, n_folds = 2, shuffle = True, random_state = 1):
    print train_indices, test_indices


# #### StratifiedKFold

# In[11]:

target = np.array([0] * 5 + [1] * 5)
print target
for train_indices, test_indices in cross_validation.StratifiedKFold(target, n_folds = 2, shuffle = True, random_state = 0):
    print train_indices, test_indices


# In[12]:

target = np.array([0, 1] * 5)
print target
for train_indices, test_indices in cross_validation.StratifiedKFold(target, n_folds = 2,shuffle = True):
    print train_indices, test_indices


# #### ShuffleSplit

# In[13]:

for train_indices, test_indices in cross_validation.ShuffleSplit(10, n_iter = 10, test_size = 0.2):
    print train_indices, test_indices


# #### StratifiedShuffleSplit

# In[14]:

target = np.array([0] * 5 + [1] * 5)
print target
for train_indices, test_indices in cross_validation.StratifiedShuffleSplit(target, n_iter = 4, test_size = 0.2):
    print train_indices, test_indices


# #### Leave-One-Out

# In[15]:

for train_indices, test_index in cross_validation.LeaveOneOut(10):
    print train_indices, test_index


# Больше стратегий проведения кросс-валидации доступно здесь: http://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators
