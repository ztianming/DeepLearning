from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import scale
import pandas as pd
import numpy as np
from sklearn import neighbors, svm, linear_model,tree, ensemble
import tensorflow as tf
import sklearn
print(tf.__version__)
print(sklearn.__version__)
print(pd.__version__)
print(np.__version__)


def accuracy(predictions, labels):
    n = len(labels)
    count=0
    for i in range(n):
        if round(predictions[i]) == labels[i]:
            count+=1
    return count/n

all_data = pd.read_excel('train.xlsx')

# 分割为训练集、验证集 比例：3：1
all_data = all_data.sample(frac=1.0)  # 全部打乱
cut_idx = int(round(0.25 * all_data.shape[0]))
validation_data, train_data = all_data.iloc[:cut_idx], all_data.iloc[cut_idx:]
# print(validation_data)
# print(train_data)

test_data = pd.read_excel('test.xlsx')
# print(train_data.head())
# print(train_data.ix[:, 1:-1])
train_x = scale(np.asarray(train_data.ix[:, 1:-1]))
train_y = np.asarray(train_data.ix[:, -1])
# print(train_x, train_y)

validation_x = scale(np.asarray(validation_data.ix[:, 1:-1]))
validation_y = np.asarray(validation_data.ix[:, -1])
# print(validation_x, validation_y)

test_x = scale(np.asarray(test_data.ix[:, 1:-1]))

# knn
k=4
knn_cls = neighbors.KNeighborsClassifier(k, weights='uniform', metric='euclidean')
predictions = knn_cls.fit(train_x, train_y).predict(validation_x)
# print(predictions)
# print(validation_y)
acc = accuracy(predictions, validation_y)
print(acc)

# svm
print("svm")
random_state = np.random.RandomState(0)
svm = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=random_state))
predictions = svm.fit(train_x, train_y).predict(validation_x)
acc = accuracy(predictions, validation_y)
print(acc)

# decision tree
print("decision tree")
dt_cls = tree.DecisionTreeClassifier()
predictions = dt_cls.fit(train_x, train_y).predict(validation_x)
acc = accuracy(predictions, validation_y)
# print(predictions)
# print(validation_y)
print(acc)


# random tree
print("random tree")
tree_num = 20
rf_cls = ensemble.RandomForestClassifier(n_estimators=tree_num)
predictions = rf_cls.fit(train_x, train_y).predict(validation_x)
acc = accuracy(predictions, validation_y)
print(acc)

# Adaboost -- not work
print("Adaboost")
tree_num = 50
ab_cls = ensemble.AdaBoostClassifier(n_estimators=tree_num)
predictions = ab_cls.fit(train_x, train_y).predict(validation_x)
acc = accuracy(predictions, validation_y)
print(acc)
# GBC
print("GBC")
gb_cls = ensemble.GradientBoostingClassifier(n_estimators=tree_num)
predictions = gb_cls.fit(train_x, train_y).predict(validation_x)
acc = accuracy(predictions, validation_y)
print(acc)

# bagging
print("Bagging")
bag_cls = ensemble.BaggingClassifier()
predictions = bag_cls.fit(train_x, train_y).predict(validation_x)
acc = accuracy(predictions, validation_y)
print(acc)

print("extraTree")
et_cls = tree.ExtraTreeClassifier()
predictions = et_cls.fit(train_x, train_y).predict(validation_x)
acc = accuracy(predictions, validation_y)
print(acc)
