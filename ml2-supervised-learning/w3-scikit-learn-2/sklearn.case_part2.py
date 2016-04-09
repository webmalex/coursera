
# coding: utf-8

# # Sklearn

# ## Bike Sharing Demand
# Задача на kaggle: https://www.kaggle.com/c/bike-sharing-demand
# 
# По историческим данным о прокате велосипедов и погодных условиях необходимо спрогнозировтаь спрос на прокат велосипедов.
# 
# В исходной псотановке задачи доступно 11 признаков: https://www.kaggle.com/c/prudential-life-insurance-assessment/data
# 
# В наборе признаков присутсвуют вещественные, категориальные, и бинарные данные. 
# 
# Для демонстрации используется обучающая выборка из исходных данных train.csv, файлы для работы прилагаются.

# ### Библиотеки

# In[1]:

from sklearn import cross_validation, grid_search, linear_model, metrics, pipeline, preprocessing

import numpy as np
import pandas as pd


# In[3]:

# %pylab inline
get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'retina'")


# ### Загрузка данных

# In[4]:

raw_data = pd.read_csv('bike_sharing_demand.csv', header = 0, sep = ',')


# In[5]:

raw_data.head()


# ### Предобработка данных

# #### Обучение и отложенный тест

# In[6]:

raw_data.datetime = raw_data.datetime.apply(pd.to_datetime)


# In[7]:

raw_data['month'] = raw_data.datetime.apply(lambda x : x.month)
raw_data['hour'] = raw_data.datetime.apply(lambda x : x.hour)


# In[8]:

train_data = raw_data.iloc[:-1000, :]
hold_out_test_data = raw_data.iloc[-1000:, :]


# In[9]:

print raw_data.shape, train_data.shape, hold_out_test_data.shape


# In[10]:

#обучение
train_labels = train_data['count'].values
train_data = train_data.drop(['datetime', 'count', 'casual', 'registered'], axis = 1)


# In[11]:

#тест
test_labels = hold_out_test_data['count'].values
test_data = hold_out_test_data.drop(['datetime', 'count', 'casual', 'registered'], axis = 1)


# In[12]:

binary_data_columns = ['holiday', 'workingday']
binary_data_indices = np.array([(column in binary_data_columns) for column in train_data.columns], dtype = bool)


# In[13]:

print binary_data_columns
print binary_data_indices


# In[14]:

categorical_data_columns = ['season', 'weather', 'month'] 
categorical_data_indices = np.array([(column in categorical_data_columns) for column in train_data.columns], dtype = bool)


# In[15]:

print categorical_data_columns
print categorical_data_indices


# In[16]:

numeric_data_columns = ['temp', 'atemp', 'humidity', 'windspeed', 'hour']
numeric_data_indices = np.array([(column in numeric_data_columns) for column in train_data.columns], dtype = bool)


# In[17]:

print numeric_data_columns
print numeric_data_indices


# ### Pipeline

# In[18]:

regressor = linear_model.SGDRegressor(random_state = 0, n_iter = 3, loss = 'squared_loss', penalty = 'l2')


# In[19]:

estimator = pipeline.Pipeline(steps = [       
    ('feature_processing', pipeline.FeatureUnion(transformer_list = [        
            #binary
            ('binary_variables_processing', preprocessing.FunctionTransformer(lambda data: data[:, binary_data_indices])), 
                    
            #numeric
            ('numeric_variables_processing', pipeline.Pipeline(steps = [
                ('selecting', preprocessing.FunctionTransformer(lambda data: data[:, numeric_data_indices])),
                ('scaling', preprocessing.StandardScaler(with_mean = 0.))            
                        ])),
        
            #categorical
            ('categorical_variables_processing', pipeline.Pipeline(steps = [
                ('selecting', preprocessing.FunctionTransformer(lambda data: data[:, categorical_data_indices])),
                ('hot_encoding', preprocessing.OneHotEncoder(handle_unknown = 'ignore'))            
                        ])),
        ])),
    ('model_fitting', regressor)
    ]
)


# In[20]:

estimator.fit(train_data, train_labels)


# In[21]:

metrics.mean_absolute_error(test_labels, estimator.predict(test_data))


# ### Подбор параметров

# In[22]:

estimator.get_params().keys()


# In[23]:

parameters_grid = {
    'model_fitting__alpha' : [0.0001, 0.001, 0,1],
    'model_fitting__eta0' : [0.001, 0.05],
}


# In[24]:

grid_cv = grid_search.GridSearchCV(estimator, parameters_grid, scoring = 'mean_absolute_error', cv = 4)


# In[25]:

get_ipython().run_cell_magic('time', '', 'grid_cv.fit(train_data, train_labels)')


# In[26]:

print grid_cv.best_score_
print grid_cv.best_params_


# ### Оценка по отложенному тесту

# In[27]:

test_predictions = grid_cv.best_estimator_.predict(test_data)


# In[28]:

metrics.mean_absolute_error(test_labels, test_predictions)


# In[29]:

print test_labels[:20]


# In[30]:

print test_predictions[:20]


# In[31]:

pylab.figure(figsize=(8, 6))
pylab.grid(True)
pylab.xlim(-100,1100)
pylab.ylim(-100,1100)
pylab.scatter(train_labels, grid_cv.best_estimator_.predict(train_data), alpha=0.5, color = 'red')
pylab.scatter(test_labels, grid_cv.best_estimator_.predict(test_data), alpha=0.5, color = 'blue')


# ### Другая модель

# In[32]:

from sklearn.ensemble import RandomForestRegressor


# In[33]:

regressor = RandomForestRegressor(random_state = 0, max_depth = 20, n_estimators = 50)


# In[34]:

estimator = pipeline.Pipeline(steps = [       
    ('feature_processing', pipeline.FeatureUnion(transformer_list = [        
            #binary
            ('binary_variables_processing', preprocessing.FunctionTransformer(lambda data: data[:, binary_data_indices])), 
                    
            #numeric
            ('numeric_variables_processing', pipeline.Pipeline(steps = [
                ('selecting', preprocessing.FunctionTransformer(lambda data: data[:, numeric_data_indices])),
                ('scaling', preprocessing.StandardScaler(with_mean = 0., with_std = 1.))            
                        ])),
        
            #categorical
            ('categorical_variables_processing', pipeline.Pipeline(steps = [
                ('selecting', preprocessing.FunctionTransformer(lambda data: data[:, categorical_data_indices])),
                ('hot_encoding', preprocessing.OneHotEncoder(handle_unknown = 'ignore'))            
                        ])),
        ])),
    ('model_fitting', regressor)
    ]
)


# In[35]:

get_ipython().run_cell_magic('time', '', 'estimator.fit(train_data, train_labels)')


# In[36]:

metrics.mean_absolute_error(test_labels, estimator.predict(test_data))


# In[38]:

test_labels[:10]


# In[39]:

estimator.predict(test_data)[:10]


# In[40]:

pylab.figure(figsize=(16, 6))

pylab.subplot(1,2,1)
pylab.grid(True)
pylab.xlim(-100,1100)
pylab.ylim(-100,1100)
pylab.scatter(train_labels, grid_cv.best_estimator_.predict(train_data), alpha=0.5, color = 'red')
pylab.scatter(test_labels, grid_cv.best_estimator_.predict(test_data), alpha=0.5, color = 'blue')
pylab.title('linear model')

pylab.subplot(1,2,2)
pylab.grid(True)
pylab.xlim(-100,1100)
pylab.ylim(-100,1100)
pylab.scatter(train_labels, estimator.predict(train_data), alpha=0.5, color = 'red')
pylab.scatter(test_labels, estimator.predict(test_data), alpha=0.5, color = 'blue')
pylab.title('random forest model')


# In[ ]:



