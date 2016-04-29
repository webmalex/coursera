
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


# In[17]:

mapping = {2 : 1, 1: 2, 0: 0}
mapped_preds = [mapping[pred] for pred in preds]
print float(sum(mapped_preds != simple_dataset.target)) / len(simple_dataset.target)


# In[18]:

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
clf = LogisticRegression()
print cross_val_score(clf, matrix, simple_dataset.target).mean()


# ## Более сложная выборка

# In[19]:

dataset = fetch_20newsgroups(
    subset='train', 
    categories=['comp.sys.mac.hardware', 'comp.os.ms-windows.misc', 'comp.graphics'])


# In[20]:

get_ipython().run_cell_magic('time', '', 'matrix = vectorizer.fit_transform(dataset.data)\nmodel = KMeans(n_clusters=3, random_state=42)\npreds = model.fit_predict(matrix.toarray())\nprint preds\nprint dataset.target')


# In[21]:

mapping = {2 : 0, 1: 1, 0: 2}
mapped_preds = [mapping[pred] for pred in preds]
print float(sum(mapped_preds != dataset.target)) / len(dataset.target)


# In[22]:

clf = LogisticRegression()
print cross_val_score(clf, matrix, dataset.target).mean()


# ## SVD + KMeans

# In[23]:

get_ipython().run_cell_magic('time', '', 'from sklearn.decomposition import TruncatedSVD\n\nmodel = KMeans(n_clusters=3, random_state=42)\nsvd = TruncatedSVD(n_components=1000, random_state=123)\nfeatures = svd.fit_transform(matrix)\npreds = model.fit_predict(features)\nprint preds\nprint dataset.target')


# In[24]:

mapping = {0 : 2, 1: 0, 2: 1}
mapped_preds = [mapping[pred] for pred in preds]
print float(sum(mapped_preds != dataset.target)) / len(dataset.target)


# In[25]:

get_ipython().run_cell_magic('time', '', 'model = KMeans(n_clusters=3, random_state=42)\nsvd = TruncatedSVD(n_components=200, random_state=123)\nfeatures = svd.fit_transform(matrix)\npreds = model.fit_predict(features)\nprint preds\nprint dataset.target')


# In[26]:

import itertools
def validate_with_mappings(preds, target, dataset):
    permutations = itertools.permutations([0, 1, 2])
    for a, b, c in permutations:
        mapping = {2 : a, 1: b, 0: c}
        mapped_preds = [mapping[pred] for pred in preds]
        print float(sum(mapped_preds != target)) / len(target)
        
validate_with_mappings(preds, dataset.target, dataset)


# In[27]:

get_ipython().run_cell_magic('time', '', 'model = KMeans(n_clusters=3, random_state=42)\nsvd = TruncatedSVD(n_components=200, random_state=321)\nfeatures = svd.fit_transform(matrix)\npreds = model.fit_predict(features)\nprint preds\nprint dataset.target\nvalidate_with_mappings(preds, dataset.target, dataset)')


# ## Итоги

# 1. Получили интерпретируемый результат на обеих выборках
# 1. Реальность, однако, намного более жестока
# 1. Попробовали использовать AgglomerativeClustering и KMeans
