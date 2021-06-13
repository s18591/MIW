import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

linearModel = 0
squareModel = 0
cubeModel = 0

def regression():
    global linearModel
    global squareModel
    global cubeModel
    daneTxt = 17
    for i in range(1,daneTxt):
        a = np.loadtxt(f"Dane/dane{i}.txt")
        x = a[:,[0]]
        y = a[:,[1]]
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
        c = np.hstack([X_train, np.ones(X_train.shape)])
        c1 = np.hstack([X_train**2, X_train, np.ones(X_train.shape)])
        c2 = np.hstack([X_train**3,X_train**2, X_train, np.ones(X_train.shape)])

        v = np.linalg.pinv(c) @ y_train
        v1 = np.linalg.pinv(c1) @ y_train
        v2 = np.linalg.pinv(c2) @ y_train
        e = sum((y_test - (v[0] * X_test + v[1])) ** 2)/len(y)

        ee = sum((y_train - (v[0] * X_train + v[1])) ** 2)/len(y_train)

        e1 = sum((y_test - (v1[0] * X_test ** 2 + v1[1] * X_test + v1[2])) ** 2)/len(y_test)

        ee1 = sum((y_train - (v1[0] * X_train ** 2 + v1[1] * X_train + v1[2])) ** 2)/len(y_train)

        e2 = sum((y_test - (v2[0] * X_test**3 + v2[1] * X_test ** 2 + v2[2] * X_test + v2[3])) ** 2)/len(y_test)

        ee2 = sum((y_train - (v2[0] * X_train**3 + v2[1] * X_train ** 2 + v2[2] * X_train + v2[3])) ** 2)/len(y_train)

        print("Test")
        if e < e1 and e < e2:
            linearModel = linearModel+1
            print(f"#{i} LINEAR")
        elif e1 < e and e1 < e2:
            squareModel = squareModel+1
            print(f"#{i} SQUARE")
        elif e2 < e and e2 < e1:
            cubeModel = cubeModel+1
            print(f"#{i} CUBE")
        print(f"ERRORS: \n LinearModel: {e}\n SquareModel: {e1}\n CubeModel: {e2}")
        print(f"linearModel: {linearModel} squareModel: {squareModel} cubeModel: {cubeModel}")

        print("Traning")
        if ee < ee1 and ee < ee2:
            linearModel = linearModel+1
            print(f"#{i} LINEAR")
        elif ee1 < ee and ee1 < ee2:
            squareModel = squareModel+1
            print(f"#{i} SQUARE")
        elif ee2 < ee and ee2 < ee1:
            cubeModel = cubeModel+1
            print(f"#{i} CUBE")
        print(f"ERRORS: \n LinearModel: {ee}\n SquareModel: {ee1}\n CubeModel: {ee2}")
        print(f"linearModel: {linearModel} squareModel: {squareModel} cubeModel: {cubeModel}")

        plt.scatter(X_test, y_test)
        plt.scatter(X_train, y_train)
        plt.plot(x, v[0] * x + v[1])
        plt.plot(x, v1[0] * x ** 2 + v1[1] * x + v1[2])
        plt.plot(x, v2[0]*x**3 + v2[1] * x ** 2 + v2[2] * x + v2[3])
       #plt.title(f"LinearModel: {e} \n SquareModel: {e1} \n CubeModel: {e2}")
        plt.show()

if __name__ == '__main__':
    regression()




