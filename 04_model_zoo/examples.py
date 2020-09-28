#!/usr/bin/python3

# load data
import sklearn.datasets
data = sklearn.datasets.load_breast_cancer()
X = data.data   # shape = (569, 30)
Y = data.target # shape = (569,)

# import classifiers
# MLP = Multilayer Perceptron = neural network
from sklearn.neural_network import MLPClassifier
mlp1 = MLPClassifier(
        hidden_layer_sizes = [100]
        )
mlp2 = MLPClassifier(
        hidden_layer_sizes = [10,10]
        )
mlp3 = MLPClassifier(
        hidden_layer_sizes = [10,10,10]
        )

from sklearn.tree import DecisionTreeClassifier
stump = DecisionTreeClassifier(
        max_depth = 1,              # k = 1, VCdim <= 2**1 = 2
        min_samples_split = 2,
        min_samples_leaf = 1,
        )
tree3 = DecisionTreeClassifier(
        max_depth = 3,              # k = 3, VCdim <= 2**3 = 8
        min_samples_split = 25,
        min_samples_leaf = 10,
        )
tree5 = DecisionTreeClassifier(
        max_depth = 5,              # k = 5, VCdim <= 2**5 = 32
        min_samples_split = 2,
        min_samples_leaf = 1,
        )

from sklearn.ensemble import AdaBoostClassifier
boosted_mlp1 = AdaBoostClassifier(
        base_estimator = mlp1,
        n_estimators = 500,
        )
boosted_mlp2 = AdaBoostClassifier(
        base_estimator = mlp2,
        n_estimators = 50,
        )
boosted_mlp3 = AdaBoostClassifier(
        base_estimator = mlp3,
        n_estimators = 11,
        )

boosted_stump = AdaBoostClassifier(
        base_estimator = stump,
        n_estimators = 50,
        )
boosted_tree3 = AdaBoostClassifier(
        base_estimator = tree3,
        n_estimators = 50,
        )
boosted_tree5 = AdaBoostClassifier(
        base_estimator = tree5,
        n_estimators = 11,
        )

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

# train model
model = mlp3
model.fit(X,Y)

# get training error
acc = model.score(X,Y)
print("acc=",acc)
