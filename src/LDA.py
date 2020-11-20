#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np


# In[6]:


reviews_datasets = pd.read_csv(r'C:\Users\sonal\Reviews.csv')
#reviews_datasets = open('Reviews.csv', 'r',errors='ignore') 


# In[7]:


reviews_datasets = reviews_datasets.head(20000)


# In[9]:


#print('reviews_datasets',reviews_datasets)
reviews_datasets.dropna()


# In[10]:


reviews_datasets.head()


# In[11]:


reviews_datasets['Text'][350]


# In[14]:


from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer(max_df=0.8, min_df=2, stop_words='english')
doc_term_matrix = count_vect.fit_transform(reviews_datasets['Text'].values.astype('U'))


# In[15]:


doc_term_matrix


# In[16]:


from sklearn.decomposition import LatentDirichletAllocation

LDA = LatentDirichletAllocation(n_components=5, random_state=42)
LDA.fit(doc_term_matrix)


# In[17]:


import random

for i in range(10):
    random_id = random.randint(0,len(count_vect.get_feature_names()))
    print(count_vect.get_feature_names()[random_id])


# In[18]:


first_topic = LDA.components_[0]


# In[19]:


top_topic_words = first_topic.argsort()[-10:]


# In[20]:


for i in top_topic_words:
    print(count_vect.get_feature_names()[i])


# In[21]:


for i,topic in enumerate(LDA.components_):
    print(f'Top 10 words for topic #{i}:')
    print([count_vect.get_feature_names()[i] for i in topic.argsort()[-10:]])
    print('\n')


# In[22]:


topic_values = LDA.transform(doc_term_matrix)
topic_values.shape


# In[23]:


reviews_datasets['Topic'] = topic_values.argmax(axis=1)


# In[24]:


reviews_datasets.head()


# In[ ]:




