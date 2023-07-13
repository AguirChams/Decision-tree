import classifier as classifier
import pandas as pd
import seaborn
import math
import random

import sns as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn import tree


df = pd.read_csv("C:/Users/PCS/Desktop/iris.csv")
print(df.head())
print(df.columns)
print(df['Species'].value_counts())
print(df.info())
print(df.sample(5))

seaborn.FacetGrid(df, hue="Species").map(plt.scatter, "Sepal.Length", "Sepal.Width").add_legend()
plt.show()
seaborn.FacetGrid(df, hue="Species").map(plt.scatter, "Petal.Length", "Petal.Width").add_legend()
plt.show()

train_set,test_set=train_test_split(df,test_size=0.3)
print(train_set.shape)
print(test_set.shape)
x_train= train_set.iloc[:,1:4]
y_train= train_set.iloc[:,5]
x_test=test_set.iloc[:,1:4]
y_test=test_set.iloc[:,5]
clf=tree.DecisionTreeClassifier(criterion="entropy",max_depth=3,max_leaf_nodes=5)
clf.fit(x_train, y_train)
tree.plot_tree(clf, filled=True)
plt.show()

prediction=clf.predict(x_test)
print(prediction)

matix = confusion_matrix(y_test,prediction)
seaborn.heatmap(matix, annot=True)
plt.show()



