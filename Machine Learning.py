# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 20:03:51 2018

@author: mzhen
"""

###Overview Machine Learning with Python - Scikit
#from sklearn.family import "Model", "Model" could mean different models
#for example, LinearRegression




USA_Housing_path = (r"C:\Users\mzhen\Desktop\Programming\Python\Python-Data-Science-and-Machine-Learning-Bootcamp\Python-Data-Science-and-Machine-Learning-Bootcamp\Machine Learning Sections\Linear-Regression\USA_Housing.csv")

df = pd.read_csv(USA_Housing_path)

df.head()
df.info()
###Linear Regression with Python part 1
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats 

df.describe()
df.columns

sns.pairplot(df)
sns.distplot(df['Price'])
sns.heatmap(df.corr(),annot=True)

#split test and training data sets
df.columns
#creating sub sets with the columns you want
X=df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]
y=df['Price']

#random_state, random split, act like a seed, so random results will be same as the video
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4
                                                    ,random_state=101)

#train
from sklearn.linear_model import LinearRegression

lm = LinearRegression()

#only fit on training datasets
lm.fit(X_train,y_train)

#print intercept

print(lm.intercept_)


#grab coff.

lm.coef_


#print all coefficient
cdf = pd.DataFrame(lm.coef_,X.columns,columns=['Coeff'])
cdf

#getting real data
from sklearn.datasets import load_boston

boston = load_boston()
boston.keys()

#print description of the data set
print(boston['DESCR'])


###Linear Regression with Python part 2
#predictions

predictions = lm.predict(X_test)
predictions

#y_test has the right price

#plot pred. to real
plt.scatter(y_test,predictions)

#plot distribution of residuals
sns.distplot((y_test-predictions))

#regression evaluation metrics (lost functions that we want to minimize)
from sklearn import metrics
#Mean Absolute Error(MAE)
metrics.mean_absolute_error(y_test,predictions)
#Mean Squared Error (MSE)
metrics.mean_squared_error(y_test,predictions)
#Root Mean Squared Error(RMSE)
np.sqrt(metrics.mean_squared_error(y_test,predictions))


#Bias Variance Trade-Off

#############################logistic modeling##########################
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

train_path = (r"C:\Users\mzhen\Desktop\Programming\Python\Python-Data-Science-and-Machine-Learning-Bootcamp\Python-Data-Science-and-Machine-Learning-Bootcamp\Machine Learning Sections\Logistic-Regression\titanic_train.csv")
test_path = (r"C:\Users\mzhen\Desktop\Programming\Python\Python-Data-Science-and-Machine-Learning-Bootcamp\Python-Data-Science-and-Machine-Learning-Bootcamp\Machine Learning Sections\Logistic-Regression\titanic_test.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

train.head()

train.isnull()

#yellow in the heatmap means null values
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

sns.set_style('whitegrid')

sns.countplot(x='Survived',data=train,hue='Sex')

sns.countplot(x='Survived',data=train,hue='Pclass')

sns.distplot(train['Age'].dropna(),kde=False,bins=30)
#or
train['Age'].plot.hist(bins=35)

train.info()

sns.countplot(x='SibSp',data=train,hue='Survived')

train['Fare'].hist(bins=40,figsize=(10,4))

#import cufflinks as cf
#cf.go_offline()
#train['Fare'].iplot(kind='hist',bins=30)

#fill in missing data in age column with avg. (imputation) by Pclass


plt.figure(figsize=(10,7))
sns.boxplot(x='Pclass',y='Age',data=train)

#get conditional means, example: df[df['W']>0]['X']

def Cond_Avg_Age_by_Pclass_inTrain(x):
    print(train[train['Pclass']==x]['Age'].mean())
    
Cond_Avg_Age_by_Pclass_inTrain(1)
Cond_Avg_Age_by_Pclass_inTrain(2) 
Cond_Avg_Age_by_Pclass_inTrain(3)


def impute_age(cols):
    Age = cols[0]
    Pclass=cols[1]
    if pd.isnull(Age):
        if Pclass==1:
            return 37 #avg. age for Pclass =1
        elif Pclass == 2:
            return 29 #avg. age for Pclass =2
        else:
            return 24 #avg. age for Pclass =3
    else:
        return Age

#fill missing with avg        
train['Age']=train[['Age','Pclass']].apply(impute_age,axis=1)

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

#drop cabin since there are too many missing values
train.drop('Cabin',axis=1,inplace=True)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

#drop any other NAs
train.dropna(inplace=True)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

#create dummy 
#male=1
sex = pd.get_dummies(train['Sex'],drop_first=True)
sex.head()

embark = pd.get_dummies(train['Embarked'],drop_first=True)
embark.head()

#add the dummies to the train data set
train = pd.concat([train,sex,embark],axis=1)
train.head()

#drop columns 
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train.head()

train.drop('PassengerId',axis=1,inplace=True)
train.head()

#model estimation only on train data since it is clean, but test data can be 
#cleaned as well

X = train.drop('Survived',axis=1)
y = train['Survived']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30,random_state=101)

from sklearn.linear_model import LogisticRegression

#fitting the model
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)

#model evaluation - classfication report
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))

#model eva. - confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)



#########################K nearest neighbours (KNN)
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


Data_path = (r"C:\Users\mzhen\Desktop\Programming\Python\Python-Data-Science-and-Machine-Learning-Bootcamp\Python-Data-Science-and-Machine-Learning-Bootcamp\Machine Learning Sections\K-Nearest-Neighbors\Classified Data")

df = pd.read_csv(Data_path,index_col=0)

df.head()

#standardizing data without knowing what they are
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS',axis=1))

scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))
scaled_features

#want all column names except the last one.
df.columns[:-1]
#now the data set is standardized
df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head()


#create train and test data sets
from sklearn.model_selection import train_test_split

X = df_feat
y = df['TARGET CLASS']

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30,random_state=101)

#fit KNN 
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)

pred = knn.predict(X_test)

#check model fitness
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))

#calculate error rate for k = 1 to k = 40
error_rate = []

for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

error_rate

#plot error rate
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue',linestyle='--'
         ,marker='o',markerfacecolor='red',markersize=10)
plt.title('Error Rate vs K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


#picking a higher k value for a lower error rate
#for example k = 17

knn=KNeighborsClassifier(n_neighbors=17)
knn.fit(X_train,y_train)
pred=knn.predict(X_test)

print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))



#Decision Trees and Random Forest
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns



Kyphosis_path = (r"C:\Users\mzhen\Desktop\Programming\Python\Python-Data-Science-and-Machine-Learning-Bootcamp\Python-Data-Science-and-Machine-Learning-Bootcamp\Machine Learning Sections\Decision-Trees-and-Random-Forests\kyphosis.csv")

df = pd.read_csv(Kyphosis_path)
df.head()
df.info()

sns.pairplot(df,hue='Kyphosis')

#split data
from sklearn.model_selection import train_test_split

X = df.drop('Kyphosis',axis=1)
y = df['Kyphosis']

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30)

from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)

predictions = dtree.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))


from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=200)

rfc.fit(X_train,y_train)

rfc_pred = rfc.predict(X_test)

print(confusion_matrix(y_test,rfc_pred))
print('\n')
print(classification_report(y_test,rfc_pred))

#absent appears to be dominate the "Kyphosis" field
#so less variation, this could affect model results
df['Kyphosis'].value_counts()







