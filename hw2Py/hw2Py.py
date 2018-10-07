import numpy as np
import numpy.linalg as lina
import pandas as pd
import matplotlib.pyplot as plt
from math import log
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

## TODO: Rename objects to indicate type
def main():
  A = Data("A")
  B = Data("B")
  C = Data("C")
  AA = gaussianMixture(A.train)
  BB = gaussianMixture(B.train)
  CC = gaussianMixture(C.train)
  #Scatter plot of basic data
  AA.estimateParameters()
  groupsList = list()
  groupsList.append(A.train.groupby('c'))
  #groupsList.append(A.test.groupby('c'))
  fig, ax = plt.subplots()
  for groups in groupsList:
    for name, group in groups:
      ax.plot(group.x, group.y, label=name, linestyle="None",marker='o')

  ##Plotting discriminant line:
  preciMat = np.linalg.inv(AA.Sigma)
  w = np.linalg.solve(AA.Sigma, AA.mu1 - AA.mu2)
  ##TODO tranpose to column vectors
  print(AA.mu1.T.dot(preciMat).dot(AA.mu1))
  w_0 = - 1/2*(AA.mu1.T.dot(preciMat).dot(AA.mu1)) \
        + 1/2*(AA.mu2.T.dot(preciMat).dot(AA.mu2)) \
        + log(AA.pi1/AA.pi2)
  x1 = [(-w[1]*x2 - w_0)/w[0] for x2 in AA.train['y']]
  plt.plot(x1, AA.train['y'])
  plt.show()
class Data():
  def __init__(self, name):
    incompletePath = "hwk2data/classification" + name

    train = pd.DataFrame(pd.read_csv(incompletePath + '.train', sep='\t'))
    test = pd.DataFrame(pd.read_csv(incompletePath + '.test', sep='\t'))
    train.columns = test.columns = ['x','y','c']
    # Joining x and y data into an array
    train['X'] = list(map(np.array, (zip(train['x'], train['y']))))
    test['X'] = list(map(np.array, (zip(test['x'], test['y']))))

    self.train = train
    self.test = test

class gaussianMixture():
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
      self.mu1 = np.transpose(sum(self.train['c']*self.train['X'])/self.N)
      self.mu2 = np.transpose(sum((1-self.train['c'])*self.train['X'])/self.N)
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

  def test(self):
    self.estimateParameters()
    cov = test.Sigma
    xx = np.linspace(-10, 10, 500)
    yy = np.linspace(-10, 10, 500)
    X,Y = np.meshgrid(xx,yy)

    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X; pos[:, :, 1] = Y
    rv1 = multivariate_normal(test.mu1, cov)
    rv2 = multivariate_normal(test.mu2, cov)
    fig, ax = plt.subplots()
    CS = ax.contour(X,Y,test.pi1*rv1.pdf(pos), levels = [0.1, 0.2, 0.3])
    CS2 = ax.contour(X,Y, test.pi2*rv2.pdf(pos))
    plt.show()

  def sample():
    rv1 = multivariate_normal(test.mu1, cov)
    rv2 = multivariate_normal(test.mu2, cov)
    pass











if __name__ == '__main__':
  main()

