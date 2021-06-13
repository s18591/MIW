import numpy as np
import matplotlib.pylab as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from plotkab import plot_decision_regions
# from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier


def main():
    iris = datasets.load_iris()
    maxDepth = 15
    ranForest = 50
    xarr = []
    yarr = []
    X = iris.data[:, [2, 3]]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    # standardyzacja cech
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    print("TREE======ENTROPY=======")
    for i in range(1, maxDepth):
        correct = 0
        wrong = 0
        tree = DecisionTreeClassifier(criterion='entropy', max_depth=i, random_state=1)
        tree.fit(X_train, y_train)
        predict = tree.predict(X)
        for k in range(len(predict)):
            if predict[k] == y[k]:
                correct = correct + 1
            else:
                wrong = wrong + 1
        xarr.append(correct)
        yarr.append(wrong)
        print(f"Depth:{i} \nCorrect: {correct} \n Wrong: {wrong} ")
    plt.plot(range(len(xarr)), xarr)
    plt.plot(range(len(yarr)), yarr)
    plt.xlabel('Correct')
    plt.ylabel('Wrong')
    plt.show()
    xarr = []
    yarr = []
    print("TREE======GINI=======")
    for i in range(1, maxDepth):
        correct = 0
        wrong = 0
        tree = DecisionTreeClassifier(criterion='gini', max_depth=i, random_state=1)
        tree.fit(X_train, y_train)
        predict = tree.predict(X)
        for k in range(len(predict)):
            if predict[k] == y[k]:
                correct = correct + 1

            else:
                wrong = wrong + 1
        yarr.append(wrong)
        xarr.append(correct)
        print(f"Depth:{i} \n Correct: {correct} \n Wrong: {wrong} ")
    plt.plot(range(len(xarr)), xarr)
    plt.plot(range(len(yarr)), yarr)
    plt.xlabel('Correct')
    plt.ylabel('Wrong')
    plt.show()
    xarr = []
    yarr = []
    # X_combined = np.vstack((X_train, X_test))
    # y_combined = np.hstack((y_train, y_test))
    # plot_decision_regions(X_combined, y_combined, classifier=tree, test_idx=range(105, 150))
    # plt.xlabel('Długość płatka [cm]')
    # plt.ylabel('Szerokość płatka [cm]')
    # plt.legend(loc='upper left')
    # plt.savefig('tree4')
    # plt.show()
    #
    # export_graphviz(tree, out_file='drzewo.dot', feature_names=['Długość płatka', 'Szerokość płatka'])

    print("FOREST======GINI=======")
    for i in range(1, ranForest):
        correct = 0
        wrong = 0
        forest = RandomForestClassifier(criterion='gini', n_estimators=i, random_state=1, n_jobs=2)
        forest.fit(X_train, y_train)
        predict = forest.predict(X)
        for k in range(len(predict)):
            if predict[k] == y[k]:
                correct = correct + 1
            else:
                wrong = wrong + 1
        xarr.append(correct)
        yarr.append(wrong)
        print(f"{i} \n Correct: {correct} \n Wrong: {wrong} ")
    plt.plot(range(len(xarr)), xarr)
    plt.plot(range(len(yarr)), yarr)
    plt.xlabel('Correct')
    plt.ylabel('Wrong')
    plt.show()
    xarr = []
    yarr = []
    # plot_decision_regions(X_combined, y_combined,classifier=forest, test_idx=range(105,150))
    # plt.xlabel('Długość płatka [cm]')
    # plt.ylabel('Szerokość płatka [cm]')
    # plt.legend(loc='upper left')
    # plt.savefig('randomforest')
    # plt.show()


if __name__ == '__main__':
    main()
