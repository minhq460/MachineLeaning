from sklearn import datasets

#import iris dataset 
iris=datasets.load_iris()

#iris.data has features, iris.target has labels
print(iris.data)
print(iris.target)

#moi entry has 4 thuoc tinh, 

#Tach du lieu thanh train và test:  test= 0.33, train = 0.66

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(iris.data,iris.target,test_size=0.33)

#import knn classifier 
from sklearn.neighbors import KNeighborsClassifier
clf=KNeighborsClassifier(n_neighbors=3).fit(x_train,y_train)

#infact knn is a lazy learner, in reality it doesnt get trained , it takes all data with it when testing each attribute


# import accuracy metric

from sklearn.metrics import accuracy_score
print("accuracy is ")
print(accuracy_score(y_test,clf.predict(x_test)))

# k=3, chính xác 96%
# thu do hoa plot voi k va accuracy


import matplotlib.pyplot as plt

#nhập vào các giá trị k khác nhau và tìm độ chính xác

## giá trị độ chính xác là mảng 2D, trong đó mỗi mục nhập là [K,accuracy]
accuracy_values=[]

for x in range(1,x_train.shape[0]):
	clf=KNeighborsClassifier(n_neighbors=x).fit(x_train,y_train)
	accuracy=accuracy_score(y_test,clf.predict(x_test))
	accuracy_values.append([x,accuracy])
	pass

# chuyển đổi mảng python bình thường thành mảng numpy để có một số hoạt động hiệu quả

import numpy as np
accuracy_values=np.array(accuracy_values)

plt.plot(accuracy_values[:,0],accuracy_values[:,1])
plt.xlabel("K")
plt.ylabel("accuracy")
plt.show()


# 40 < K < 60 thì accuracy tốt 
