
# coding: utf-8

# In[74]:

from pandas import read_csv
from sklearn.cluster import MeanShift
import numpy as np
from io import StringIO
import sys
sys.path.append('../..')
from lib import *
from collections import Counter


# In[14]:

df = read_csv('checkins.dat','|', skiprows=[1], na_values='                   ').dropna() 
#, , header=1, , index_col='id', skipfooter=2
df.columns = [s.strip() for s in df.columns]
df.head()


# In[15]:

# 396632
df.info()


# In[36]:

# with open('checkins.dat', 'r') as f: print(f.read(500))
print(open('checkins.dat').read(700))


# In[47]:

X = df[['latitude','longitude']].sample(100000)
X.info()


# In[48]:

ms = MeanShift(bandwidth=0.1, n_jobs=-1)
get_ipython().magic('time ms.fit(X)')


# In[49]:

# labels = ms.labels_
# cluster_centers = ms.cluster_centers_
# labels_unique = np.unique(labels)
# n_clusters_ = len(labels_unique)
# print("number of estimated clusters : %d" % n_clusters_)
# print(cluster_centers)
cc = ms.cluster_centers_
cc


# In[69]:

cl = ms.labels_


# In[70]:

len(np.unique(cl))


# In[56]:

len(cc)


# In[138]:

of = [tuple(o) for o in np.array(read_csv(StringIO('''
33.751277, -118.188740, (Los Angeles)
25.867736, -80.324116, (Miami)
51.503016, -0.075479, (London)
52.378894, 4.885084, (Amsterdam)
39.366487, 117.036146, (Beijing)
-33.868457, 151.205134, (Sydney)
'''), header=None))[:,0:2]]
of


# In[68]:

from geopy.distance import vincenty
newport_ri = (41.49008, -71.312796)
cleveland_oh = (41.499498, -81.695391)
print(vincenty(newport_ri, cleveland_oh).meters)


# In[67]:

import pickle
def pd(f, a):
    with open(f, 'wb') as handle:
        pickle.dump(a, handle)

def pl(f):
    with open(f, 'rb') as handle:
        return pickle.load(handle)
pd('ms.p', ms)


# In[83]:

# np.array(Counter(cl).items())
# %%time
cl1 = list(cl)
get_ipython().magic('time cl2 = np.array([[i,cl1.count(i)] for i in set(cl)])')


# In[94]:

cl3 = cl2[cl2[:,1] > 15]


# In[107]:

vincenty((of[0,0], of[0,1]), (of[1,0], of[1,1])).meters


# In[111]:

cl4 = cl3[:,0]


# In[137]:

# cc1 = cc[cl4]
cc2 = [(i,tuple(cc[i])) for i in cl4]
# cc2


# In[144]:

# v = lambda o,c: vincenty(tuple(o), tuple(c)).meters 
# %time b = [(v(o, c), i) for o in of for i,c in enumerate(cc1)]
b = [(vincenty(o, c).meters, i) for o in of for i,c in cc2]
b.sort()


# In[151]:

# -33.8661460662 151.207082415
pf('m3-w1',pp(cc[b[0][1]]))


# In[152]:

pf('m3-w1-1',pp(cc[b[1][1]]))


# In[ ]:



