import numpy as np
import numpy.linalg as lina
import pandas as pd
import matplotlib.pyplot as plt
from math import log
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D


def main():
  gaussData = generativeModel()
  for i ,gauss in enumerate(gaussData):
    gaussData[i].plotContours()
  plt.show()

def logisticRegression():
  pass
def generativeModel():
  data = list()
  gausData = list()
  #The class Data is initialized by passing the relevant string
  for name in ["A", "B", "C"]:
    data.append(Data(name))

  #Every instance of a model and its data have their own class
  for i, dat in enumerate(data):
    gausData.append(GaussianMixture(dat.train))
    gausData[i].estimateParameters()

  #Scatter plot of the test data on different figures
  for i, dat in enumerate(data):
    groups = dat.test.groupby('c')
    for name, group in groups:
        plt.figure(i)
        plt.plot(group.x, group.y, label=name,
                 linestyle="None",marker='o',alpha=0.6)

  #Computation of the boundary decision and plotting it on relevant fig
  for i, gauss in enumerate(gausData):
    preciMat = np.linalg.inv(gauss.Sigma)
    w = np.linalg.solve(gauss.Sigma, gauss.mu1 - gauss.mu2)
    #quadratic Forms in einstein notation
    w_0 = - 1/2*np.einsum("s,st,t->",gauss.mu1, preciMat, gauss.mu1)\
          + 1/2*np.einsum("s,st,t->",gauss.mu2, preciMat, gauss.mu2)\
          + log(gauss.pi1/gauss.pi2)
    #compute the line p(C_1 | x) = 0.5 for x as a function of y
    x1 = [(-w[1]*x2 - w_0)/w[0] for x2 in gauss.train['y']]
    plt.figure(i)
    plt.plot(x1, gauss.train['y'])
    plt.draw()

  return gausData

class Data():
  def __init__(self, name):
    #We fetch the data only according to its "letter" name
    incompletePath = "hwk2data/classification" + name

    train = pd.DataFrame(pd.read_csv(incompletePath + '.train', sep='\t'))
    test = pd.DataFrame(pd.read_csv(incompletePath + '.test', sep='\t'))
    train.columns = test.columns = ['x','y','c']
    # We join x and y data into a numpy array
    train['X'] = train[['x','y']].values.tolist()
    test['X'] = test[['x','y']].values.tolist()

    self.train = train
    self.test = test
  def saveToCsv():
    pass

class GaussianMixture():
  def __init__(self, train):
    self.train = train
    self.parametersEstimated = False
  def estimateParameters(self):
    def estHyper(self):
      self.N1 = sum(self.train['c'])
      self.N2 = sum(1 - self.train['c'])
      self.N = self.N1 + self.N2
      self.pi1 = self.N1/self.N
      self.pi2 = self.N2/self.N
    def estMean(self):
      self.mu1 = sum([c*np.array(X) for c,X in \
                      zip(self.train['c'], self.train['X'])])/self.N
      self.mu2 = sum([(1-c)*np.array(X) for c,X in \
                      zip(self.train['c'], self.train['X'])])/self.N
    def estVariance(self):
      X = self.train['X']
      C = self.train['c']
      sumPart_1 = sum([c * np.outer(x - self.mu1, x - self.mu1) for x,c  in zip(X,C)])
      sumPart_2 = sum([(1-c) * np.outer(x - self.mu2, x - self.mu2) for x,c in zip(X,C)])
      self.Sigma = (sumPart_1 + sumPart_2)/self.N
    if(not self.parametersEstimated):
      estHyper(self)
      estMean(self)
      estVariance(self)
      self.parametersEstimated = True
  def plotContours(self):
    self.estimateParameters()
    cov = self.Sigma
    xx = np.linspace(-10, 10, 500)
    yy = np.linspace(-10, 10, 500)
    X,Y = np.meshgrid(xx,yy)

    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X; pos[:, :, 1] = Y
    rv1 = multivariate_normal(self.mu1, self.Sigma)
    rv2 = multivariate_normal(self.mu2, self.Sigma)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.contour(X,Y, self.pi1*rv1.pdf(pos))
    ax.contour(X,Y, self.pi2*rv2.pdf(pos))
  def sample():
    rv1 = multivariate_normal(test.mu1, cov)
    rv2 = multivariate_normal(test.mu2, cov)
    pass


class LogisticRegression():
  def __ini__(self,train):
    self.train = train
    self.parametersEstimated = False

  def irls(self):
    X = np.stack(train['X'])

    def __sigmoid(z):
      return 1 / (1 + np.exp(-z))

    def __sig_of_linear(w,x):
      return __sigmoid(w.T.dot(x))

    def __weightingMatrix(self,w):
      x = self.train['X']
      sig_of_linear = __sig_of_linear(w,x)
      return np.diag([sig_of_linear * (1 - sig_of_linear)])

    def __zVector(w,self) :
      R = __weightingMatrix(w)
      Y = __sigmoid(w.dot(X).T)
      z = X.dot(w) - np.linalg.solve(R, Y - train['c'])
      return z

    w_init = np.array([0,0])






if __name__ == '__main__':
  main()

