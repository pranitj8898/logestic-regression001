# logestic-regression001

#import libs
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#import dataset
titanic_data=pd.read_csv('titanic_train.csv')

titanic_data.head()
titanic_data.describe()

#Fill the missing values we will fill the missing values for age. In order to fill missing values we use fillna method. For now we will fill the missing age by taking average of all age.

titanic_data['Age'].fillna(titanic_data['Age'].mean(),inplace=True)
titanic_data['Age'].isna().sum()

#We can see cabin column has a number of null values, as such we can not use it for prediction. Hence we will drop it

titanic_data.drop('Cabin',axis=1,inplace=True)

#drop sex
gender=pd.get_dummies(titanic_data['Sex'],drop_first=True)
titanic_data['Gender']=gender
titanic_data.head()

#Seperate Dependent and Independent variables

x=titanic_data[['PassengerId','Pclass','Age','SibSp','Parch','Fare','Gender']]
y=titanic_data['Survived']

#Regression model

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)

#prediction and accurcy
predict=lr.predict(x_test)
from sklearn.metrics import confusion_matrix
pd.DataFrame(confusion_matrix(y_test,predict),columns=['Predicted No','Predicted Yes'],index=['Actual No','Actual Yes'])
from sklearn.metrics import classification_report
print(classification_report(y_test,predict))

