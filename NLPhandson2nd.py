#!/usr/bin/env python
# coding: utf-8

# In[24]:


#session 


# In[25]:


from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(binary = True)


# In[26]:


corpus = ["The Mercedes which picks me up from the Delhi airport feels exorbitant. The bottom of my jeans is beginning to tear and the flight has crumpled my shirt. Walking into the marble lobby of the Oberoi, New Delhi, I feel typically discrepant, but the welcome I receive is warm. I think I even recognise the doorman. His smile is intimate; his greeting, familiar. I have been here before. No, I am not an impostor.","Interrupted only by two months of torrential downpour, Bombay suffers valiantly its 300 days of summer. Usually dripping with either rain or sweat, we certainly know what it is like to be drenched. Weather doesn’t make for good banter here. Like bad conversation, our seasons too are made of monosyllables. Parts of the country might have it marginally better, but let’s face it—as a whole, we Indians sadly need to be privileged to enjoy summer. We need to be able to turn on our air conditioners.","Depending on their destiny, adventurers are either fools or heroes of their stories. I have no illusions as to which one I am. In my writings elsewhere, I have professed as much, admitting to more folly than valour. I storm in when caution is called for, swing big and miss bigger. But even during my most colossal missteps, I never lack chutzpah. In a more reflective light, my bravado is as entertaining as it is sobering."]


# In[27]:


vect.fit(corpus)


# In[28]:


vocab = vect.vocabulary_


# In[29]:


for key in vocab.keys():
    print("{}:{}".format(key,vocab[key]))


# In[30]:


print(vect.transform(["The Mercedes which picks me up from the Delhi airport feels exorbitant. The bottom of my jeans is beginning to tear and the flight has crumpled my shirt. Walking into the marble lobby of the Oberoi, New Delhi, I feel typically discrepant, but the welcome I receive is warm. I think I even recognise the doorman. His smile is intimate; his greeting, familiar. I have been here before. No, I am not an impostor."]).toarray())


# In[31]:


from sklearn.metrics.pairwise import cosine_similarity
similarity=cosine_similarity(vect.transform(["Interrupted only by two months of torrential downpour, Bombay suffers valiantly its 300 days of summer. Usually dripping with either rain or sweat, we certainly know what it is like to be drenched. Weather doesn’t make for good banter here. Like bad conversation, our seasons too are made of monosyllables. Parts of the country might have it marginally better, but let’s face it—as a whole, we Indians sadly need to be privileged to enjoy summer. We need to be able to turn on our air conditioners."]).toarray(),vect.transform(["The Mercedes which picks me up from the Delhi airport feels exorbitant. The bottom of my jeans is beginning to tear and the flight has crumpled my shirt. Walking into the marble lobby of the Oberoi, New Delhi, I feel typically discrepant, but the welcome I receive is warm. I think I even recognise the doorman. His smile is intimate; his greeting, familiar. I have been here before. No, I am not an impostor."]).toarray())


# In[32]:


print(similarity)


# In[ ]:





# In[ ]:




