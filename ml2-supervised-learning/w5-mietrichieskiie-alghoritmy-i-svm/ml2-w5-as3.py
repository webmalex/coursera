
# coding: utf-8

# In[16]:

from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.ensemble import RandomForestClassifier as RFC
import sys
sys.path.append('../..')
from lib import *


# In[9]:

ds = load_digits()
X = ds.data
y = ds.target
g = int(len(y)*0.75)
Xl = X[:g]
Xt = X[g:]
yl = y[:g]
yt = y[g:]
g


# In[7]:

print(X[:1], y[:20])


# In[18]:

get_ipython().magic("time pf('1', 1-KNC(1).fit(Xl,yl).score(Xt,yt))")


# In[17]:

get_ipython().magic("time pf('2', 1-RFC(1000).fit(Xl,yl).score(Xt,yt))")


# In[ ]:



