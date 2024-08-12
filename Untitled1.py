#!/usr/bin/env python
# coding: utf-8

# In[50]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from warnings import filterwarnings
filterwarnings(action='ignore')


# In[51]:


pd.set_option('display.max_columns',10,'display.width',1000)
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
train.head()


# In[52]:


train.shape


# In[53]:


test.shape


# In[54]:


train.isnull().sum()


# In[55]:


test.isnull().sum()


# In[56]:


train.describe(include="all")


# In[57]:


train.groupby('Survived').mean()


# In[58]:


train.corr()


# In[59]:


male_ind=len(train[train['Sex']=='male'])
print("No of Males in Titanic:",male_ind)


# In[60]:


female_ind=len(train[train['Sex']=='female'])
print("No of Females in Titanic:",female_ind)


# In[61]:


fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
gender=['Male','Female']
index=[577,314]
ax.bar(gender,index)
plt.xlabel("Gender")
plt.ylabel("No of people onboarding ship")
plt.show()


# In[62]:


alive=len(train[train['Survived']==1])
dead=len(train[train['Survived']==0])


# In[63]:


train.groupby('Sex')[['Survived']].mean()


# In[64]:


fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
status=['Survived','Dead']
ind=[alive,dead]
ax.bar(status,ind)
plt.xlabel("Status")
plt.show()


# In[65]:


plt.figure(1)
train.loc[train['Survived']==1,'Pclass'].value_counts().sort_index().plot.bar()
plt.title('Bar graph of people accrding to ticket class in which people survived')

plt.figure(2)
train.loc[train['Survived']==0,'Pclass'].value_counts().sort_index().plot.bar()
plt.title('Bar graph of people accrding to ticket class in which people couldn\'t survive')


# In[66]:


plt.figure(1)
age =train.loc[train.Survived==1,'Age']
plt.title('The histogram of the age groups of the people that had survived')
plt.hist(age, np.arange(0,100,10))
plt.xticks(np.arange(0,100,10))
plt.figure(2)
age =train.loc[train.Survived==0, 'Age']
plt.title('The histogram of the age groups of the people that coudn\'t survive')
plt.hist(age, np.arange(0,100,10))
plt.xticks(np.arange(0,100,10))


# In[67]:


train[["SibSp","Survived"]].groupby(['SibSp'],as_index=False).mean().sort_values(by='Survived',ascending=False)


# In[68]:


train[["Pclass","Survived"]].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived',ascending=False)


# In[69]:


train[["Age","Survived"]].groupby(['Age'],as_index=False).mean().sort_values(by='Age',ascending=True)


# In[70]:


train[["Embarked","Survived"]].groupby(['Embarked'],as_index=False).mean().sort_values(by='Survived',ascending=False)


# In[71]:


fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
ax.axis('equal')
l=['C=Cherbourg','Q=Queenstown','S=Southampton']
s = [0.553571,0.389610,0.336957]
ax.pie(s, labels=l,autopct='%1.2f%%')
plt.show()


# In[72]:


test.describe(include="all")


# In[73]:


train=train.drop(['Ticket'],axis=1)
test=test.drop(['Ticket'],axis=1)


# In[74]:


train=train.drop(['Cabin'],axis=1)
test=test.drop(['Cabin'],axis=1)


# In[75]:


train=train.drop(['Name'],axis=1)
test=test.drop(['Name'],axis=1)


# In[76]:


column_train=['Age','Pclass','SibSp','Parch','Fare','Sex','Embarked']
X=train[column_train]
Y=train['Survived']


# In[77]:


X['Age'].isnull().sum()
X['Pclass'].isnull().sum()
X['SibSp'].isnull().sum()
X['Parch'].isnull().sum()
X['Fare'].isnull().sum()
X['Sex'].isnull().sum()
X['Embarked'].isnull().sum()


# In[78]:


X['Age']=X['Age'].fillna(X['Age'].median())
X['Age'].isnull().sum()


# In[79]:


d={'male':0,'female':1}
X['Sex']=X['Sex'].apply(lambda x:d[x])
X['Sex'].head()


# In[80]:


X['Embarked']=X['Embarked'].fillna('C')  
X['Embarked']=X['Embarked'].apply(lambda x: e[x])


# In[81]:


unexpected_values=set(X['Embarked'])-set(e.keys())
print(unexpected_values)


# In[82]:


X['Embarked']=X['Embarked'].map(e).fillna(-1)


# In[83]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=7)


# In[84]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train,Y_train)
Y_pred=model.predict(X_test)
from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(Y_test,Y_pred))


# In[85]:


from sklearn.metrics import accuracy_score,confusion_matrix
confusion_mat=confusion_matrix(Y_test,Y_pred)
print(confusion_mat)


# In[86]:


from sklearn.neighbors import KNeighborsClassifier
model2=KNeighborsClassifier(n_neighbors=5)
model2.fit(X_train,Y_train)
y_pred2=model2.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(Y_test,y_pred2))


# In[87]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
confusion_mat=confusion_matrix(Y_test,y_pred2)
print(confusion_mat)
print(classification_report(Y_test,y_pred2))


# In[88]:


from sklearn.naive_bayes import GaussianNB
model3=GaussianNB()
model3.fit(X_train,Y_train)
y_pred3=model3.predict(X_test)
from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(Y_test,y_pred3))


# In[89]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
confusion_mat=confusion_matrix(Y_test,y_pred3)
print(confusion_mat)
print(classification_report(Y_test,y_pred3))


# In[90]:


from sklearn.tree import DecisionTreeClassifier
model4=DecisionTreeClassifier(criterion='entropy',random_state=7)
model4.fit(X_train,Y_train)
y_pred4=model4.predict(X_test)
from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(Y_test,y_pred4))


# In[91]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
confusion_mat=confusion_matrix(Y_test,y_pred4)
print(confusion_mat)
print(classification_report(Y_test,y_pred4))


# In[92]:


results=pd.DataFrame({
    'Model': ['Logistic Regression','Support Vector Machines','Naive Bayes','KNN','Decision Tree'],
    'Score': [0.75,0.66,0.76,0.66,0.74]})
result_df=results.sort_values(by='Score', ascending=False)
result_df=result_df.set_index('Score')
result_df.head(9)


# In[ ]:




