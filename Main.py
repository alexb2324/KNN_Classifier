import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn import model_selection


class KNN:
    def __init__(self, X, y):
        self.data = X
        self.targets = y

    def distance(self, X):
        sample_num, _ = X.shape
        arr = []
        for i in range(sample_num):
            arr.append(np.sqrt(np.sum((self.data - X[i]) ** 2, axis = 1)))

        return np.array(arr)

    def predict(self, X, k):

        dist = self.distance(X)

        knn = np.argsort(dist)[:, :k]
        y_vals = self.targets[knn]
        if k == 1:
            return y_vals.T
        else:
            sample_num, _ = X.shape
            arr = []
            for i in range(sample_num):
                arr.append(max(y_vals[i], key = list(y_vals[i]).count))
            return arr


def knn_train(X, y):

    test_scores = []
    for _ in range(5):
        temp = []
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y)
        knn = KNN(X_train, y_train)

        test = knn.predict(X_test, k = 1)
        test_acc = np.sum(test[0] == y_test) / len(test[0]) * 100
        temp.append(test_acc)

        for i in range(2, 6):
            test = knn.predict(X_test, i)
            test_acc = np.sum(test == y_test) / len(test) * 100
            temp.append(test_acc)

        test_scores.append(temp)

    test_scores = np.array(test_scores)
    test_scores_average = np.mean(test_scores, axis=0)
    print("Highest accuracy for at", str(round(test_scores.max(), 2)) + "%")
    print("Highest average accuracy with k =", np.argmax(test_scores_average), "at", str(round(np.max(test_scores_average), 2)) + "%")


def main():
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target

    knn_train(X, y)


if __name__ == "__main__":
    main()