
# coding: utf-8

# # Пример: кластеризация текстов

# ## Выборка

# In[1]:

get_ipython().run_cell_magic('time', '', "from sklearn.datasets import fetch_20newsgroups\n\ntrain_all = fetch_20newsgroups(subset='train')\nprint train_all.target_names")


# In[2]:

simple_dataset = fetch_20newsgroups(
    subset='train', 
    categories=['comp.sys.mac.hardware', 'soc.religion.christian', 'rec.sport.hockey'])


# In[3]:

print simple_dataset.data[0]


# In[4]:

simple_dataset.target


# In[5]:

print simple_dataset.data[-1]


# In[6]:

print simple_dataset.data[-2]


# In[7]:

print len(simple_dataset.data)


# ## Признаки

# In[8]:

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

vectorizer = TfidfVectorizer(max_df=500, min_df=10)
matrix = vectorizer.fit_transform(simple_dataset.data)


# In[9]:

print matrix.shape


# ## Аггломеративная кластеризация (neighbour joining)

# In[10]:

get_ipython().run_cell_magic('time', '', "from sklearn.cluster.hierarchical import AgglomerativeClustering\n\nmodel = AgglomerativeClustering(n_clusters=3, affinity='cosine', linkage='complete')\npreds = model.fit_predict(matrix.toarray())")


# In[11]:

print list(preds)


# In[12]:

print matrix[0]


# In[168]:

vectorizer.get_feature_names()


# In[13]:

vectorizer.get_feature_names()[877]


# In[14]:

simple_dataset.data[0]


# 
# ## KMeans

# In[15]:

get_ipython().run_cell_magic('time', '', 'from sklearn.cluster import KMeans\n\nmodel = KMeans(n_clusters=3, random_state=1)\npreds = model.fit_predict(matrix.toarray())\nprint preds')


# In[16]:

print simple_dataset.target


# In[174]:

mapping = {2 : 1, 1: 2, 0: 0}
mapped_preds = [mapping[pred] for pred in preds]
print float(sum(mapped_preds != simple_dataset.target)) / len(simple_dataset.target)


# In[175]:

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
clf = LogisticRegression()
print cross_val_score(clf, matrix, simple_dataset.target).mean()


# ## Более сложная выборка

# In[176]:

dataset = fetch_20newsgroups(
    subset='train', 
    categories=['comp.sys.mac.hardware', 'comp.os.ms-windows.misc', 'comp.graphics'])


# In[177]:

matrix = vectorizer.fit_transform(dataset.data)
model = KMeans(n_clusters=3, random_state=42)
preds = model.fit_predict(matrix.toarray())
print preds
print dataset.target


# In[178]:

mapping = {2 : 0, 1: 1, 0: 2}
mapped_preds = [mapping[pred] for pred in preds]
print float(sum(mapped_preds != dataset.target)) / len(dataset.target)


# In[179]:

clf = LogisticRegression()
print cross_val_score(clf, matrix, dataset.target).mean()


# ## SVD + KMeans

# In[180]:

from sklearn.decomposition import TruncatedSVD

model = KMeans(n_clusters=3, random_state=42)
svd = TruncatedSVD(n_components=1000, random_state=123)
features = svd.fit_transform(matrix)
preds = model.fit_predict(features)
print preds
print dataset.target


# In[181]:

mapping = {0 : 2, 1: 0, 2: 1}
mapped_preds = [mapping[pred] for pred in preds]
print float(sum(mapped_preds != dataset.target)) / len(dataset.target)


# In[182]:

model = KMeans(n_clusters=3, random_state=42)
svd = TruncatedSVD(n_components=200, random_state=123)
features = svd.fit_transform(matrix)
preds = model.fit_predict(features)
print preds
print dataset.target


# In[183]:

import itertools
def validate_with_mappings(preds, target, dataset):
    permutations = itertools.permutations([0, 1, 2])
    for a, b, c in permutations:
        mapping = {2 : a, 1: b, 0: c}
        mapped_preds = [mapping[pred] for pred in preds]
        print float(sum(mapped_preds != target)) / len(target)
        
validate_with_mappings(preds, dataset.target, dataset)


# In[184]:

model = KMeans(n_clusters=3, random_state=42)
svd = TruncatedSVD(n_components=200, random_state=321)
features = svd.fit_transform(matrix)
preds = model.fit_predict(features)
print preds
print dataset.target
validate_with_mappings(preds, dataset.target, dataset)


# ## Итоги

# 1. Получили интерпретируемый результат на обеих выборках
# 1. Реальность, однако, намного более жестока
# 1. Попробовали использовать AgglomerativeClustering и KMeans
