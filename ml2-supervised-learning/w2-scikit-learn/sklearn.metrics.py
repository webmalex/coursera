
# coding: utf-8

# # Sklearn

# ## sklearn.metrics

# документация: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics

# In[1]:

from sklearn import cross_validation, datasets, linear_model, metrics 
from matplotlib.colors import ListedColormap


# In[2]:

get_ipython().magic('pylab inline')


# ### Генерация датасетов

# In[3]:

clf_data, clf_target = datasets.make_classification(n_features = 2, n_informative = 2, n_classes = 2, 
                                                    n_redundant = 0, n_clusters_per_class = 1, 
                                                    random_state = 7)


# In[4]:

reg_data, reg_target = datasets.make_regression(n_features = 2, n_informative = 1, n_targets = 1, 
                                                noise = 5., random_state = 7)


# In[5]:

colors = ListedColormap(['red', 'blue'])
pylab.scatter(map(lambda x: x[0], clf_data), map(lambda x: x[1], clf_data), c = clf_target, cmap = colors)


# In[6]:

pylab.scatter(map(lambda x:x[1], reg_data), reg_target, color = 'r')
pylab.scatter(map(lambda x:x[0], reg_data), reg_target, color = 'b')


# In[7]:

clf_train_data, clf_test_data, clf_train_labels, clf_test_labels = cross_validation.train_test_split(clf_data, clf_target,
                                                                                     test_size = 0.3, random_state = 1)


# In[8]:

reg_train_data, reg_test_data, reg_train_labels, reg_test_labels = cross_validation.train_test_split(reg_data, reg_target,
                                                                                     test_size = 0.3, random_state = 1)


# ### Метрики качества в задачах классификации

# #### Обучение модели классификации

# In[9]:

classifier = linear_model.SGDClassifier(loss = 'log', random_state = 1)


# In[10]:

classifier.fit(clf_train_data, clf_train_labels)


# In[11]:

predictions = classifier.predict(clf_test_data)


# In[12]:

probability_predictions = classifier.predict_proba(clf_test_data)


# In[13]:

print clf_test_labels


# In[14]:

print predictions


# In[15]:

print probability_predictions


# #### accuracy

# In[16]:

sum([1. if pair[0] == pair[1] else 0. for pair in zip(clf_test_labels, predictions)])/len(clf_test_labels)


# In[17]:

metrics.accuracy_score(clf_test_labels, predictions)


# #### confusion matrix

# In[18]:

matrix = metrics.confusion_matrix(clf_test_labels, predictions)
print matrix


# In[19]:

sum([1 if pair[0] == pair[1] else 0 for pair in zip(clf_test_labels, predictions)])


# In[20]:

matrix.diagonal().sum()


# #### precision 

# In[21]:

metrics.precision_score(clf_test_labels, predictions, pos_label = 0)


# In[22]:

metrics.precision_score(clf_test_labels, predictions)


# #### recall

# In[23]:

metrics.recall_score(clf_test_labels, predictions, pos_label = 0)


# In[24]:

metrics.recall_score(clf_test_labels, predictions)


# #### f1

# In[25]:

metrics.f1_score(clf_test_labels, predictions, pos_label = 0)


# In[26]:

metrics.f1_score(clf_test_labels, predictions)


# #### classification report

# In[27]:

print metrics.classification_report(clf_test_labels, predictions)


# #### ROC curve

# In[28]:

fpr, tpr, _ = metrics.roc_curve(clf_test_labels, probability_predictions[:,1])


# In[29]:

pylab.plot(fpr, tpr, label = 'linear model')
pylab.plot([0, 1], [0, 1], '--', color = 'grey', label = 'random')
pylab.xlim([-0.05, 1.05])
pylab.ylim([-0.05, 1.05])
pylab.xlabel('False Positive Rate')
pylab.ylabel('True Positive Rate')
pylab.title('ROC curve')
pylab.legend(loc = "lower right")


# #### ROC AUC

# In[30]:

metrics.roc_auc_score(clf_test_labels, predictions)


# In[31]:

metrics.roc_auc_score(clf_test_labels, probability_predictions[:,1])


# #### PR AUC

# In[32]:

metrics.average_precision_score(clf_test_labels, predictions)


# #### log_loss

# In[33]:

metrics.log_loss(clf_test_labels, probability_predictions[:,1])


# ### Метрики качества в задачах регрессии

# #### Обучение регрессионной модели 

# In[34]:

regressor = linear_model.SGDRegressor(random_state = 1, n_iter = 20)


# In[35]:

regressor.fit(reg_train_data, reg_train_labels)


# In[36]:

reg_predictions = regressor.predict(reg_test_data)


# In[37]:

print reg_test_labels


# In[38]:

print reg_predictions


# #### mean absolute error

# In[39]:

metrics.mean_absolute_error(reg_test_labels, reg_predictions)


# #### mean squared error

# In[40]:

metrics.mean_squared_error(reg_test_labels, reg_predictions)


# #### root mean squared error

# In[41]:

sqrt(metrics.mean_squared_error(reg_test_labels, reg_predictions))


# #### r2 score

# In[42]:

metrics.r2_score(reg_test_labels, reg_predictions)

