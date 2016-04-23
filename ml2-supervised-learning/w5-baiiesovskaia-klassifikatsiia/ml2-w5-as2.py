
# coding: utf-8

# In[40]:

from sklearn import datasets
from sklearn.cross_validation import cross_val_score
from sklearn import naive_bayes as nb
# import numpy as np
# %matplotlib inline
import sys
sys.path.append('../..')
from lib import *


# In[24]:

digits = datasets.load_digits()


# In[6]:

breast_cancer = datasets.load_breast_cancer()


# In[26]:

X1 = digits.data
y1 = digits.target
X2 = breast_cancer.data
y2 = breast_cancer.target


# In[19]:

print(digits.data[:1], digits.target[:10])


# In[22]:

print(breast_cancer.data[:1], breast_cancer.target[:30])


# In[39]:

def test1(t, est, X, y):
    print(t, cross_val_score(est, X, y).mean())
for est in [nb.BernoulliNB(), nb.MultinomialNB(), nb.GaussianNB()]:
    print(est)
    for ds in [digits, breast_cancer]:
        test1(ds.DESCR[:10], est, ds.data, ds.target)


# In[45]:

pf('1', '0.936749280609')
pf('2', '0.870877148974')
pf('3', '3 4')


# In[44]:

print(breast_cancer.DESCR)


# In[ ]:



