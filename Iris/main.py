
#import libraries
import warnings
warnings.filterwarnings('ignore')
import numpy as np #for linear algebra
import pandas as pd #for data preprocessing
import matplotlib.pyplot as plt #for visulaize your result
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
test=[]
test1=[]
print("Enter the input features of a Iris Flower")
print("Petal Length")
p_l=float(input())
test.append(p_l)
print("Petal Width")
p_w=float(input())
test.append(p_w)
print("Sepal Length")
s_l=float(input())
test.append(s_l)
print("Sepal Width")
s_w=float(input())
test.append(s_w)
test1.append(test)

dataset=pd.read_csv("Iris.csv")
X=dataset.iloc[:,1:5].values
y=dataset.iloc[:,5].values
label=LabelEncoder()
y=label.fit_transform(y)
encoder=OneHotEncoder(sparse=False)
y=encoder.fit_transform(y.reshape(-1,1))
sc=StandardScaler()
X=sc.fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)*100
#print(accuracy)
y_result=classifier.predict(test1)
if(y_result[0][0]==1.0):
	print("Species:-> Iris-setosa")
elif(y_result[0][1]==1.0):
	print("Species:-> Iris-versicolor")
elif(y_result[0][2]==1.0):
	print("Species:-> Iris-virginica")
else:
	print("Wrong data")


print("accuracy:-")
print(accuracy)