# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Prepare your data Clean and format your data Split your data into training and testing sets

2.Define your model Use a sigmoid function to map inputs to outputs Initialize weights and bias terms

3.Define your cost function Use binary cross-entropy loss function Penalize the model for incorrect predictions

4.Define your learning rate Determines how quickly weights are updated during gradient descent

5.Train your model Adjust weights and bias terms using gradient descent Iterate until convergence or for a fixed number of iterations

6.Evaluate your model Test performance on testing data Use metrics such as accuracy, precision, recall, and F1 score

7.Tune hyperparameters Experiment with different learning rates and regularization techniques

8.Deploy your model Use trained model to make predictions on new data in a real-world application.


## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: SANGAVI SURESH
RegisterNumber:  212222230130
*/
import pandas as pd
df=pd.read_csv("/content/Employee.csv")

df.head()

df.info()

df.isnull().sum()

df["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

df["salary"]=le.fit_transform(df["salary"])
df.head()

x=df[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=df["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```
*/
## Output:
# INITIAL DATA SET:
![image](https://github.com/Sangavi-suresh/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118541861/e6d32b78-9a6e-407d-b345-522f49f9baca)

# DATA INFO:
![image](https://github.com/Sangavi-suresh/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118541861/43615de8-5cf9-4f8c-8c22-d344fb21c58f)

# OPTIMIZATION OF NULL VALUES:
![image](https://github.com/Sangavi-suresh/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118541861/d492d6f2-d9ea-43d5-92e5-87892bd91d56)

# ASSIGNMENT OF X AND Y VALUES:
![image](https://github.com/Sangavi-suresh/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118541861/679b9298-2870-4e22-b7a9-2a7eed469abc)

# CONVERTING STRING LITERALS TO NUMERICAL VALUES USING LABEL ENCODER:
![image](https://github.com/Sangavi-suresh/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118541861/d1b84344-fa4a-4257-9dbe-8a5c5b618145)

# ACCURACY:
![image](https://github.com/Sangavi-suresh/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118541861/d3b0dadc-65b0-43a2-888d-6190f62a940f)

# PREDICTION:
![image](https://github.com/Sangavi-suresh/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118541861/dafe83d1-9b36-4fa1-863f-b75d5a4c508b)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
