import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

def main():
  A = Data("A")
  B = Data("B")
  C = Data("C")
  AA = gaussianMixture(A.train)
  BB = gaussianMixture(B.train)
  CC = gaussianMixture(C.train)

  print(AA.mu1, AA.mu2, AA.Sigma)
class Data():
  def __init__(self, name):
    incompletePath = "hwk2data/classification" + name

    train = pd.DataFrame(pd.read_csv(incompletePath + '.train', sep='\t'))
    test = pd.DataFrame(pd.read_csv(incompletePath + '.test', sep='\t'))
    train.columns = test.columns = ['x','y','c']
    # Joining x and y data into an array
    train['X'] = list(map(np.array, (zip(train['x'], train['y']))))
    test['X'] = list(map(np.array, (zip(test['x'], test['y']))))
    train.drop(columns=['x','y'], inplace=True)
    test.drop(columns=['x','y'], inplace=True)

    self.train = train
    self.test = test
class gaussianMixture():
  def __init__(self, train):
    self.train = train
    self.estimateParameters()
  def estimateParameters(self):
    self.estHyper()
    self.estMean()
    self.estVariance()
  def estHyper(self):
    self.N1 = sum(self.train['c'])
    self.N2 = sum(1 - self.train['c'])
    self.N = self.N1 + self.N2
    self.pi1 = self.N1/self.N
    self.pi2 = self.N2/self.N
  def estMean(self):
    self.mu1 = sum(self.train['c']*self.train['X'])/self.N
    self.mu2 = sum((1-self.train['c'])*self.train['X'])/self.N
  def estVariance(self):
    X = self.train['X']
    C = self.train['c']
    sumPart_1 = sum([c * np.outer(x - self.mu1, x - self.mu1) for x,c  in zip(X,C)])
    sumPart_2 = sum([(1-c) * np.outer(x - self.mu2, x - self.mu2) for x,c in zip(X,C)])
    self.Sigma = (sumPart_1 + sumPart_2)/self.N
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













if __name__ == '__main__':
  main()

