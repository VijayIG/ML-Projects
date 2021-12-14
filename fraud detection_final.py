#!/usr/bin/env python
# coding: utf-8

# # LogisticRegression

# In[1]:


import pandas as pd
# import seaborn as sb
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv("payment_fraud.csv")
df


# In[3]:


df.info()


# In[4]:


df.describe()


# In[5]:


df.isnull().sum()


# In[6]:


# plt.figure(figsize=(15,5))
# # sb.heatmap(df.isnull(),cbar=False,yticklabels=False,cmap="magma");


# In[7]:


# from scipy.stats import norm,skewnorm


# In[8]:


# sb.catplot(x="paymentMethod",y="localTime",hue="paymentMethod",kind="boxen",data=df)


# In[9]:


# sb.distplot(df.paymentMethodAgeDays,bins=5,color="r");


# In[10]:


# sb.barplot(x="paymentMethod",y="localTime",hue="paymentMethod",data=df)


# In[11]:


# sb.scatterplot(x="paymentMethodAgeDays",y="accountAgeDays",data=df,color="navy");


# In[12]:


# sb.countplot(df.label)


# In[13]:


# a=sb.distplot(df.accountAgeDays,bins=6,kde_kws={"lw":3},hist_kws={"histtype":"stepfilled","alpha":1},color="r")


# In[14]:


# sb.set(style="white",color_codes=True)
# b=sb.FacetGrid(df,col='paymentMethod')
# b.map(plt.hist,"label");


# In[15]:


# sb.pairplot(data=df);


# In[16]:


from sklearn.preprocessing import LabelEncoder
enco=LabelEncoder()
df["paymentMethod"]=enco.fit_transform(df.paymentMethod)
df


# In[17]:


fraud=df[df["label"]==1]
valid=df[df["label"]==0]


# In[18]:


valid.shape,fraud.shape


# In[19]:


x_ind=df.drop("label",axis=1)


# In[20]:


y_dep=df.label


# In[21]:


x_ind


# In[22]:


y_dep


# In[23]:


from sklearn import preprocessing
x_scaled=preprocessing.scale(x_ind)
x_scaled


# In[24]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
x_train,x_test,y_train,y_test=train_test_split(x_scaled,y_dep,test_size=0.2,random_state=3)


# In[25]:


x_train.shape,x_test.shape,y_train.shape,y_test.shape


# In[26]:


classifier=LogisticRegression()


# In[27]:


classifier.fit(x_train,y_train)


# In[28]:


y_pred_log=classifier.predict(x_test)


# In[29]:


y_pred_log


# In[30]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred_log)


# In[32]:


acc=classifier.score(x_test,y_pred_log)
print(acc)


# In[33]:


from sklearn.metrics import accuracy_score


# In[34]:


accuracy_score(y_test,y_pred_log)


# In[35]:


print("Number of mislabeled points out of a total %d points : %d"
      % (x_test.shape[0], (y_test != y_pred_log).sum()))


# In[ ]:





# In[36]:


import pickle


# In[37]:


pickle_out=open("classifier.pkl","wb")


# In[38]:


pickle.dump(classifier,pickle_out)
pickle_out.close()


# In[ ]:




