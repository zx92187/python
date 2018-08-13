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



##############Decision Trees and Random Forest
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


##########Support Vector Machines (SVM)
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
cancer.keys()
#get details of the data set
print(cancer['DESCR'])

#grab features
df_feat=pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
df_feat.info()

cancer['target']
#what does target mean
cancer['target_names']

from sklearn.model_selection import train_test_split

X=df_feat
y=cancer['target']


X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30)


from sklearn.svm import SVC

model = SVC()
model.fit(X_train,y_train)

predictions = model.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

#report says everything belows to a single group, model failed
#need to search for the right parameters for the model
print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))

#grid_search
from sklearn.model_selection import GridSearchCV

#trying to find the right parameters
#below, 'C' controls the cost of mis-classification, 
    #the larger C gives lower bias but higher variance
#gamma is free parameter in the radio basis function (kernel = 'rbf')
    #larger gamma means higher bias and low vairance 
    
#SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#  decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
#  max_iter=-1, probability=False, random_state=None, shrinking=True,
#  tol=0.001, verbose=False)

#create the parameters we are going to do grid search on
param_grid = {'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}

#need to check meaning of verbose 
grid=GridSearchCV(SVC(),param_grid,verbose=3)
grid.fit(X_train,y_train)

#get best parameters and estimators
grid.best_params_
grid.best_estimator_
                  
grid_predictions = grid.predict(X_test)                  
              
print(confusion_matrix(y_test,grid_predictions))
print('\n')
print(classification_report(y_test,grid_predictions))



###########################K Means Clustering for unlabeled the data
#unsupervised learning - not to predict any particular variable
#we are looking for patterns
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

#creating a fake data set with certain features
from sklearn.datasets import make_blobs
data = make_blobs(n_samples=200,n_features=2,centers=4,cluster_std=1.8,
                  random_state=101)

#200 data points with 2 features (variables)
data[0].shape

plt.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4)
kmeans.fit(data[0])

kmeans.cluster_centers_
kmeans.labels_
#if you do not know the labels, at stage, you are done.

#since our data is unsupervised we the labels
#plot with matplotlib
fig , (ax1,ax2) = plt.subplots(1,2, sharey=True,figsize=(10,6))
#plot predicted labels
ax1.set_title('K means')
ax1.scatter(data[0][:,0],data[0][:,1],c=kmeans.labels_,cmap='rainbow')
#plot actual labels
ax2.set_title('Original')
ax2.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')
#in the plots, colours do not mean the same in two plots
#we can change number of clusters in kmeans = KMeans(n_clusters=4)

####Principle Compoent Analysis 
#dimension reduction by forming combinations of the features
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

type(cancer)
cancer.keys()
#get descr of the data
print(cancer['DESCR'])

df = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
df.head()

#check the labeled data
cancer['target']
#and check the labels 
cancer['target_names']

#standardize the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df)
scaled_data=scaler.transform(df)

#PCA
from sklearn.decomposition import PCA

pca=PCA(n_components=2)
pca.fit(scaled_data)

x_pca=pca.transform(scaled_data)
#the original scaled data have 30 dimensions
scaled_data.shape
#the pca fitted data have 2 dimensions because we defined
#n_components=2 above
x_pca.shape

plt.figure(figsize=(8,6))
#c means color, this shows out of the 30 dimensions 
#we have isolated 2 dimensions that clearly divide the labeled variable
#in this case, 'target'
plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'])
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')

#try to understand
pca.components_
df_comp = pd.DataFrame(pca.components_,columns=cancer['feature_names'])
df_comp

plt.figure(figsize=(12,6))
sns.heatmap(df_comp,cmap='plasma')

###Recommender Systems
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('white')

Data_path = (r"C:\Users\mzhen\Desktop\Programming\Python\Python-Data-Science-and-Machine-Learning-Bootcamp\Python-Data-Science-and-Machine-Learning-Bootcamp\Machine Learning Sections\Recommender-Systems\u.data")
Item_path = (r"C:\Users\mzhen\Desktop\Programming\Python\Python-Data-Science-and-Machine-Learning-Bootcamp\Python-Data-Science-and-Machine-Learning-Bootcamp\Machine Learning Sections\Recommender-Systems\u.item")
Movie_Id_Path = (r"C:\Users\mzhen\Desktop\Programming\Python\Python-Data-Science-and-Machine-Learning-Bootcamp\Python-Data-Science-and-Machine-Learning-Bootcamp\Machine Learning Sections\Recommender-Systems\Movie_Id_Titles")

columns_names = ['user_id','item_id','rating','timestamp']

df = pd.read_csv(Data_path,sep='\t',names=columns_names)
df.head()

movie_titles = pd.read_csv(Movie_Id_Path)
movie_titles.head()

df = pd.merge(df,movie_titles,on='item_id')
df.head()

#check movies with highest ratings regardless of number of ratings received
df.groupby('title')['rating'].mean().sort_values(ascending=False).head()

#check movies with most ratings
df.groupby('title')['rating'].count().sort_values(ascending=False).head()

#creat a ratings dataframe
#step 1
ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings.head()

#step 2 add number of ratings
ratings['num of ratings']=pd.DataFrame(df.groupby('title')['rating'].count())
ratings.head()

#step 3 visualize number of ratings
ratings['num of ratings'].hist(bins=70)
#dist of rating of movies
ratings['rating'].hist(bins=70)
#dist b/w avg rating and number of ratings
sns.jointplot(x='rating',y='num of ratings',data=ratings,alpha=0.5)

#crating recommender system based on item similarity
df.head()

#creating matrix
moviemat=df.pivot_table(index='user_id',columns='title',values='rating')
moviemat.head()

ratings.sort_values('num of ratings',ascending=False).head(10)

starwars_user_ratings = moviemat['Star Wars (1977)']
liarliar_user_ratings = moviemat['Liar Liar (1997)']

#compute pair-wise correlation of two data set 
similar_to_starwars = moviemat.corrwith (starwars_user_ratings)
similar_to_liarliar = moviemat.corrwith (liarliar_user_ratings)

#getting similar movies - recommender system
corr_starwars=pd.DataFrame(similar_to_starwars,columns=['Correlation'])
corr_starwars.dropna(inplace=True)
corr_starwars.head()

#the strange ones that are perfectly correlated because these movies 
#have little ratings 
corr_starwars.sort_values('Correlation',ascending=False).head(10)

#set a threshold on number of ratings so that we can filter out
#movies with less reviews

#filter out any movies with <100 reviews
#step 1 add num of ratings to data
corr_starwars = corr_starwars.join(ratings['num of ratings'])
corr_starwars.head()
#filter
corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation',ascending=False).head()

corr_liarliar = pd.DataFrame(similar_to_liarliar,columns=['Correlation'])
corr_liarliar
corr_liarliar.dropna(inplace=True)

corr_liarliar=corr_liarliar.join(ratings['num of ratings'])
corr_liarliar[corr_liarliar['num of ratings']>100].sort_values('Correlation',ascending=False).head()


#Natual Language Processing
###build spam detector
import nltk
#downloading data
#nltk.download_shell()

Data_path = (r"C:\Users\mzhen\Desktop\Programming\Python\Python-Data-Science-and-Machine-Learning-Bootcamp\Python-Data-Science-and-Machine-Learning-Bootcamp\Machine Learning Sections\Natural-Language-Processing\smsspamcollection\SMSSpamCollection")

messages = [line.rstrip() for line in open(Data_path)]
print(len(messages))
#check random messages
messages[50]

for mess_no,message in enumerate(messages[:10]):
    print(mess_no,message)
    print('\n')
    
messages[0]
#'ham\tGo until jurong point, crazy.. 
#Available only in bugis n great world la e buffet... Cine there got amore wat...'
#"\t" means tab separated

#transform the raw text file into a data file
import pandas as pd

messages = pd.read_csv(Data_path,sep='\t',names=['label','message'])

messages.head()

#check unique values
messages.describe()

#check unique values by group
messages.groupby('label').describe()

#creat a new variable with variable length
messages['length']=messages['message'].apply(len)
messages.head()

#visualize length
import matplotlib.pyplot as plt
import seaborn as sns

messages['length'].plot.hist(bins=50)

messages['length'].describe()

#grab the longest message in the data and print the entire msg
messages[messages['length']==910]['message'].iloc[0]

#check histograms of ham and spam message length 
messages.hist(column='length',by='label',bins=60,figsize=(12,4))
