import numpy as np
import matplotlib.pyplot as plt

def relo():
 P = np.arange(-2, 2.1, 0.1).reshape(1, len(np.arange(-2, 2.1, 0.1)))
 T = P ** 2 + (np.random.rand(1, len(P[0, :]))-0.5)
 #siec
 s1 = 100
 w1 = np.random.rand(s1, 1)-0.5
 b1 = np.random.rand(s1, 1)-0.5
 w2 = np.random.rand(1, s1)-0.5
 b2 = np.random.rand(1, 1)-0.5
 lr = 0.001

 #plt.plot(P, T, 'r*')
 # plt.plot(P, a2)
 plt.ion()
 fig = plt.figure()
 ax = fig.add_subplot(1, 1, 1)
 line1, = ax.plot(P[0, :], np.zeros(len(P[0, :])))
 ax.plot(P, T, 'r*')


 for epoka in range(1, 1000):
  #x = w1*P + b1*np.ones(len(P[:, 0]))
  a1 = np.tanh(w1@P + b1@np.ones(P.shape))
  a2 = w2@a1 + b2

  #propagacja wsteczna
  e2 = T - a2
  e1 = w2.T @ e2

  dW2 = lr * e2 @ a1.T
  dB2 = lr * e2 @ (np.ones(e2.shape)).T
  dW1 = lr * (1-a1*a1) * e1 @ P.T
  dB1 = lr * (1-a1*a1) * e1 @ (np.ones(P.shape)).T

  w2 = w2 + dW2
  b2 = b2 + dB2
  w1 = w1 + dW1
  b1 = b1 + dB1
  line1.set_ydata(a2.reshape(a2.shape[1]))
  fig.canvas.draw()
  fig.canvas.flush_events()


if __name__ == '__main__':
 relo()
 plt.ioff()
 plt.show()