
# coding: utf-8

# In[32]:


#importing the libraries
#3 essential library
#numpy constis of mathematical functions,
#matplotlib for plotting graphs and plots
#pandas for importing dataset and data managing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import seaborn as sns
sns.set(color_codes=True)


#import the dataset using pandas
dataset=pd.read_csv('Iris_data.csv')
dataset.head()

#Summary of the dataset
#Shape(gives number of rows and no.ofcolumns in the dataset)
print(dataset.shape)

#Info(gives datatype,constraints of each column in the dataset)
print(dataset.info())

#Descriptions(gives descriptive statistics like mean,min etc of each column)
print(dataset.describe())

#Class Dirtibution using GROUPBY(gives distribution of each type of classification over whole dataset)
print(dataset.groupby('SpeciesType').size())

#Visualizations:
#Box and Whisker Plots for all cloumns in the dataset
dataset.plot(kind='box' , sharex=False, sharey=False)

#Histograms
dataset.hist(edgecolor='black', linewidth=1.2)

#Boxplot on each feature split out by Class
dataset.boxplot(by="SpeciesType",figsize=(10,10))

#Scatter Plot Matrix
from pandas.plotting import scatter_matrix
scatter_matrix(dataset,figsize=(10,10))
plt.show()

#bivariate relation between each pair of features using seaborn pairplot
sns.pairplot(dataset, hue="SpeciesType")

#importing metrics for evalutaions
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#matrix of features,Independent variables(X)
X=dataset.iloc[:,:-1].values

X

#matrix of features,dependent variables(Y)
Y=dataset.iloc[:,4].values

Y

#replacing missing data using library imputer class from skleran 
#imputer replaces missing values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(X[:,2:4])
X[:,2:4]=imputer.transform(X[:,2:4])

X

#incase of dependent variable,only label encoder is sufficient and no need of onehotencoder as python will consider that they are dependent hence already in category
from sklearn.preprocessing import LabelEncoder
labelencoder_Y=LabelEncoder()
Y=labelencoder_Y.fit_transform(Y)

Y

#splitting the dataset into training and test set
#randon_state='num' always generate same results otherwise it will generate different result everytym
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

#Models
#LogisticRegression
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(X_train,Y_train)

#predicting the test set results
y_pred=classifier.predict(X_test)

#Summary of the predictions made by the classifier
print(classification_report(Y_test, y_pred))
print(confusion_matrix(Y_test, y_pred))

# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(Y_test,y_pred))

#Naive Bayes
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, Y_train)

y_pred = classifier.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(Y_test, y_pred))
print(confusion_matrix(Y_test, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(Y_test,y_pred))

# Support Vector Machine's 
from sklearn.svm import SVC
classifier = SVC()
classifier.fit(X_train, Y_train)

y_pred = classifier.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(Y_test, y_pred))
print(confusion_matrix(Y_test, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(Y_test,y_pred))


# K-Nearest Neighbours
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=8)
classifier.fit(X_train, Y_train)

y_pred = classifier.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(Y_test, y_pred))
print(confusion_matrix(Y_test, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(Y_test,y_pred))

# Decision Tree's
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, Y_train)

y_pred = classifier.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(Y_test, y_pred))
print(confusion_matrix(Y_test, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,Y_test))


