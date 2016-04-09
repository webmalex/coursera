
# coding: utf-8

# # Sklearn

# ## Bike Sharing Demand
# Задача на kaggle: https://www.kaggle.com/c/bike-sharing-demand
# 
# По историческим данным о прокате велосипедов и погодным условиям необходимо оценить спрос на прокат велосипедов.
# 
# В исходной постановке задачи доступно 11 признаков: https://www.kaggle.com/c/prudential-life-insurance-assessment/data
# 
# В наборе признаков присутсвуют вещественные, категориальные, и бинарные данные. 
# 
# Для демонстрации используется обучающая выборка из исходных данных train.csv, файлы для работы прилагаются.

# ### Библиотеки

# In[1]:

from sklearn import cross_validation, grid_search, linear_model, metrics

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


# ***datetime*** - hourly date + timestamp  
# 
# ***season*** -  1 = spring, 2 = summer, 3 = fall, 4 = winter 
# 
# ***holiday*** - whether the day is considered a holiday
# 
# ***workingday*** - whether the day is neither a weekend nor holiday
# 
# ***weather*** - 1: Clear, Few clouds, Partly cloudy, Partly cloudy
# 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
# 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
# 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog 
#     
# ***temp*** - temperature in Celsius
# 
# ***atemp*** - "feels like" temperature in Celsius
# 
# ***humidity*** - relative humidity
# 
# ***windspeed*** - wind speed
# 
# ***casual*** - number of non-registered user rentals initiated
# 
# ***registered*** - number of registered user rentals initiated
# 
# ***count*** - number of total rentals

# In[6]:

print raw_data.shape


# In[7]:

raw_data.isnull().values.any()


# ### Предобработка данных

# #### Типы признаков

# In[8]:

raw_data.info()


# In[9]:

raw_data.datetime = raw_data.datetime.apply(pd.to_datetime)


# In[10]:

raw_data['month'] = raw_data.datetime.apply(lambda x : x.month)
raw_data['hour'] = raw_data.datetime.apply(lambda x : x.hour)


# In[11]:

raw_data.head()


# #### Обучение и отложенный тест

# In[12]:

train_data = raw_data.iloc[:-1000, :]
hold_out_test_data = raw_data.iloc[-1000:, :]


# In[13]:

print raw_data.shape, train_data.shape, hold_out_test_data.shape


# In[14]:

print 'train period from {} to {}'.format(train_data.datetime.min(), train_data.datetime.max())
print 'evaluation period from {} to {}'.format(hold_out_test_data.datetime.min(), hold_out_test_data.datetime.max())


# #### Данные и целевая функция

# In[15]:

#обучение
train_labels = train_data['count'].values
train_data = train_data.drop(['datetime', 'count'], axis = 1)


# In[16]:

#тест
test_labels = hold_out_test_data['count'].values
test_data = hold_out_test_data.drop(['datetime', 'count'], axis = 1)


# #### Целевая функция на обучающей выборке и на отложенном тесте

# In[17]:

pylab.figure(figsize = (16, 6))

pylab.subplot(1,2,1)
pylab.hist(train_labels)
pylab.title('train data')

pylab.subplot(1,2,2)
pylab.hist(test_labels)
pylab.title('test data')


# #### Числовые признаки

# In[18]:

numeric_columns = ['temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'month', 'hour']


# In[19]:

train_data = train_data[numeric_columns]
test_data = test_data[numeric_columns]


# In[20]:

train_data.head()


# In[21]:

test_data.head()


# ### Модель

# In[22]:

regressor = linear_model.SGDRegressor(random_state = 0)


# In[23]:

regressor.fit(train_data, train_labels)
metrics.mean_absolute_error(test_labels, regressor.predict(test_data))


# In[24]:

print test_labels[:10]


# In[25]:

print regressor.predict(test_data)[:10]


# In[26]:

regressor.coef_


# ### Scaling

# In[27]:

from sklearn.preprocessing import StandardScaler


# In[28]:

#создаем стандартный scaler
scaler = StandardScaler()
scaler.fit(train_data, train_labels)
scaled_train_data = scaler.transform(train_data)
scaled_test_data = scaler.transform(test_data)


# In[29]:

regressor.fit(scaled_train_data, train_labels)
metrics.mean_absolute_error(test_labels, regressor.predict(scaled_test_data))


# In[30]:

print test_labels[:10]


# In[31]:

print regressor.predict(scaled_test_data)[:10]


# ### Подозрительно хорошо?

# In[32]:

print regressor.coef_


# In[33]:

print map(lambda x : round(x, 2), regressor.coef_)


# In[34]:

train_data.head()


# In[35]:

train_labels[:10]


# In[36]:

np.all(train_data.registered + train_data.casual == train_labels)


# In[37]:

train_data.drop(['casual', 'registered'], axis = 1, inplace = True)
test_data.drop(['casual', 'registered'], axis = 1, inplace = True)


# In[38]:

scaler.fit(train_data, train_labels)
scaled_train_data = scaler.transform(train_data)
scaled_test_data = scaler.transform(test_data)


# In[39]:

regressor.fit(scaled_train_data, train_labels)
metrics.mean_absolute_error(test_labels, regressor.predict(scaled_test_data))


# In[40]:

print map(lambda x : round(x, 2), regressor.coef_)


# ### Pipeline

# In[41]:

from sklearn.pipeline import Pipeline


# In[42]:

#создаем pipeline из двух шагов: scaling и классификация
pipeline = Pipeline(steps = [('scaling', scaler), ('regression', regressor)])


# In[43]:

pipeline.fit(train_data, train_labels)
metrics.mean_absolute_error(test_labels, pipeline.predict(test_data))


# ### Подбор параметров

# In[44]:

pipeline.get_params().keys()


# In[45]:

parameters_grid = {
    'regression__loss' : ['huber', 'epsilon_insensitive', 'squared_loss', ],
    'regression__n_iter' : [3, 5, 10, 50], 
    'regression__penalty' : ['l1', 'l2', 'none'],
    'regression__alpha' : [0.0001, 0.01],
    'scaling__with_mean' : [0., 0.5],
}


# In[46]:

grid_cv = grid_search.GridSearchCV(pipeline, parameters_grid, scoring = 'mean_absolute_error', cv = 4)


# In[47]:

get_ipython().run_cell_magic('time', '', 'grid_cv.fit(train_data, train_labels)')


# In[48]:

print grid_cv.best_score_
print grid_cv.best_params_


# ### Оценка по отложенному тесту

# In[49]:

metrics.mean_absolute_error(test_labels, grid_cv.best_estimator_.predict(test_data))


# In[50]:

np.mean(test_labels)


# In[51]:

test_predictions = grid_cv.best_estimator_.predict(test_data)


# In[52]:

print test_labels[:10]


# In[53]:

print test_predictions[:10]


# In[54]:

pylab.figure(figsize=(16, 6))

pylab.subplot(1,2,1)
pylab.grid(True)
pylab.scatter(train_labels, pipeline.predict(train_data), alpha=0.5, color = 'red')
pylab.scatter(test_labels, pipeline.predict(test_data), alpha=0.5, color = 'blue')
pylab.title('no parameters setting')
pylab.xlim(-100,1100)
pylab.ylim(-100,1100)

pylab.subplot(1,2,2)
pylab.grid(True)
pylab.scatter(train_labels, grid_cv.best_estimator_.predict(train_data), alpha=0.5, color = 'red')
pylab.scatter(test_labels, grid_cv.best_estimator_.predict(test_data), alpha=0.5, color = 'blue')
pylab.title('grid search')
pylab.xlim(-100,1100)
pylab.ylim(-100,1100)


# In[ ]:



