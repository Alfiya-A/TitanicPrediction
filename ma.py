import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


df=pd.read_csv('/Users/alfia/Desktop/Mainn/titanic.csv')

df.drop(['PassengerId','Name','Ticket','Cabin'],axis=1, inplace=True)

le_ma = LabelEncoder()
le_em = LabelEncoder()

df['Age']=df['Age'].fillna(df.Age.median())

df.Age = df.Age.apply(int)

df.Embarked = df.Embarked.fillna('S')

df['Gender']=le_ma.fit_transform(df['Sex'])
df['Embark']=le_em.fit_transform(df['Embarked'])

df.drop(['Sex','Embarked'],axis=1,inplace=True)

x=df[['Pclass','Gender','Age','SibSp','Parch','Fare','Embark']]
y=df.Survived

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train,y_train)

y_pred = lr.predict(x_test)

print(lr.predict([[3,1,22,1,0,7.2500,2]]))