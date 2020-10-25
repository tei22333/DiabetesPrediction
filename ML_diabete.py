# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 15:48:14 2020

@author: Yumped
"""
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix
df=pd.read_csv("diabetes.csv")

#axis=1 means collumn
x = df.drop("Outcome",axis=1).values
#outcome data
y = df['Outcome'].values

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.4)

#find best k
k_neig = np.arange(1,9)
train_score = np.empty(len(k_neig))
test_score = np.empty(len(k_neig))
# unlock comment by ctrl + 5
# =============================================================================
best_score = 0.0
for i,k in enumerate(k_neig):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    train_score[i] = knn.score(x_train,y_train)
    test_score[i] = knn.score(x_test,y_test)
    best_score = max(test_score[i],best_score)
print(best_score)
plt.title("comparison")
plt.plot(k_neig,test_score,label="Test score")
plt.plot(k_neig,train_score,label="Train score")
plt.legend()
plt.xlabel("k_num")
plt.ylabel("Score")
plt.show()
# =============================================================================
# =============================================================================
# knn = KNeighborsClassifier(n_neighbors=7)
# knn.fit(x_train,y_train)
# y_pred=knn.predict(x_test)
# print(classification_report(y_test,y_pred))
# print(confusion_matrix(y_test,y_pred))
# =============================================================================
