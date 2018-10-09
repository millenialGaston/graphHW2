import numpy as np
import numpy.linalg as lina
import pandas as pd
import matplotlib.pyplot as plt
from math import log
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import fsolve
import os
import errno



#TODO make seperate figures for different algos
#For the moment run as interactive if more
#tha one algo is ran, this will plot all
#decision boundaries on the same scatter plots
#of the data but not overwrite the ones saved
#for tex generation

def main():

  interactive = True

  pathForFigures = "../texhwk2/figures/"
  directory = "./dump/"
  if interactive:
    pathForFigures = createFolder(directory)

  plotMe = True
  driver = Driver(pathForFigures)
  driver.scatter()
  driver.computeGenerativeModel(plotMe)
  driver.computeLogisticRegression(plotMe)
  driver.computeLinearRegression(plotMe)
  driver.computeQDA()
  if(plotMe):
    plt.show()

class Driver():
  def __init__(self, pathForFigures):

    self.data = list()
    self.genData = list()
    self.logData = list()
    self.linData = list()
    self.qdaData = list()
    self.pathForFigures = pathForFigures

    #The class Data is initialized by passing the relevant string
    for name in ["A", "B", "C"]:
      self.data.append(Data(name))

    #Overwrite old values since the file will be opened in append mode 
    f = open("MissRates","w")
    f.close()

  #Scatter plot of the test data on different figures
  def scatter(self):
    for i, dat in enumerate(self.data):
      groups = dat.test.groupby('c')
      markers = ['o', 'x']
      for (name, group), marker in zip(groups, markers):
          fig = plt.figure(i)
          plt.plot(group.x, group.y, label=name,
                   linestyle="None",marker=marker,alpha=0.6)

  def computeGenerativeModel(self, plotMe):
    for i, dat in enumerate(self.data):
      self.genData.append(LDA(dat.train,dat.test))
      self.genData[i].estimateParameters()

    #Computation of the boundary decision and plotting it on relevant fig
    with open("MissRates", "a") as f:
      f.write("Generative Model miss rate : \n")
      for i, gen in enumerate(self.genData):

        #miss rate
        missRate = gen.computeMisclassificationRate()
        f.write(str(missRate) + "\n")

        #plotting
        if(plotMe):
          fig = plt.figure(i)
          x1 = gen.computeBoundary()
          plt.plot(x1, gen.test['y'] , label="Fisher LDA")
          plt.legend()
          if os.path.exists(self.pathForFigures):
            fig.savefig(self.pathForFigures + 'generativeFig' + str(i))
          else:
            fig.savefig('generativeFig' + str(i))

  def computeLogisticRegression(self,plotMe):
    for i, dat in enumerate(self.data):
      self.logData.append(LogisticRegression(dat.train,dat.test))
      self.logData[i].estimateParameters()

    with open("MissRates", "a") as f:
      f.write("Logistic Regression Misclassification Rate: \n")
      for i, logis in enumerate(self.logData):

        #missRate
        missRate = logis.computeMisclassificationRate()
        f.write(str(missRate) + "\n")

        if(plotMe):
          x1 = logis.computeBoundary()
          fig = plt.figure(i)
          plt.plot(x1,logis.test['y'], label="Logistic Regression")
          plt.legend()
          if os.path.exists(self.pathForFigures):
            fig.savefig(self.pathForFigures + 'logisticRegression' + str(i))
          else:
            fig.savefig('LogisticRegression' + str(i))

  def computeLinearRegression(self,plotMe):
    for i, dat in enumerate(self.data):
      self.linData.append(LinearRegression(dat.train,dat.test))
      self.linData[i].estimateParameters()
    with open("MissRates", "a") as f:
      f.write("Linear Regression Misclassification Rate: \n")
      for i, line in enumerate(self.linData):

        missRate = line.computeMisclassificationRate()
        f.write(str(missRate) + "\n")

        if(plotMe):
          x1 = line.computeBoundary()
          fig = plt.figure(i)
          plt.plot(x1,line.test['y'], label="Linear Regression")
          plt.legend()
          if os.path.exists(self.pathForFigures):
            fig.savefig(self.pathForFigures + 'LinearRegression' + str(i))
          else:
            fig.savefig('LinearRegression' + str(i))

  def computeQDA(self):
    for i, dat in enumerate(self.data):
      self.qdaData.append(QDA(dat.train,dat.test))
      self.qdaData[i].estimateParameters()
    with open("MissRates", "a") as f:
      f.write("QDA Missclassification Rate: \n")
      for i, quad in enumerate(self.qdaData):
        missRate = quad.computeMisclassificationRate()
        f.write(str(missRate) + "\n")

class LDA():
  def __init__(self, train, test):
    self.train = train
    self.test = test
    self.parametersEstimated = False
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
  def computeLinearParameters(self):
    self.estimateParameters()
    preciMat = np.linalg.inv(self.Sigma)
    w = np.linalg.solve(self.Sigma, self.mu1 - self.mu2)
    #quadratic Forms in einstein notation
    w_0 = - 1/2*np.einsum("s,st,t->",self.mu1, preciMat, self.mu1)\
          + 1/2*np.einsum("s,st,t->",self.mu2, preciMat, self.mu2)\
          + log(self.pi1/self.pi2)
    return (w_0,w)

  #Common Interface
  def estimateParameters(self):
    def estHyper(self):
      self.N1 = sum(self.train['c'])
      self.N2 = sum(1 - self.train['c'])
      self.N = self.N1 + self.N2
      self.pi1 = self.N1/self.N
      self.pi2 = self.N2/self.N
    def estMean(self):
      self.mu1 = sum([c*np.array(X) for c,X in \
                      zip(self.train['c'], self.train['X'])])/self.N1
      self.mu2 = sum([(1-c)*np.array(X) for c,X in \
                      zip(self.train['c'], self.train['X'])])/self.N2
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

  def computeBoundary(self):
    (w_0, w) = self.computeLinearParameters()
    print(w_0,w)
    #compute the line p(C_1 | x) = 0.5 for x as a function of y
    x1 = [(-w[1]*x2 - w_0)/w[0] for x2 in self.test['y']]
    return x1

  def computeMisclassificationRate(self):
    (w_0,w) = self.computeLinearParameters()
    w = np.hstack([w_0, w])
    for index, row in self.test.iterrows():
      x_with_bias = np.hstack([1, row['X']])
      pC_1lx = sigmoid(x_with_bias @ w)
      self.test.loc[index, 'GOOD']  = (pC_1lx > 0.5) and row['c'] ==1 or (pC_1lx <= 0.5)

    rate = ((self.test.shape[0] - sum(self.test['GOOD']))/self.test.shape[0])
    return rate

class LogisticRegression():
  def __init__(self,train,test):
    self.train = train
    self.test = test
    self.parametersEstimated = False
    self.wList = list()
    self.iterations = 7
    self.param = 0
    X = np.stack(self.train['X'])
    a = np.ones(len(X))
    a.shape = (len(a),1)
    self.X = np.hstack((a,X))

  def computeY(self,w):
    linear = self.X @ w
    return sigmoid(linear)
  def iterate(self,w) :
    w.shape = (len(w),1)
    Y = self.computeY(w)[:,0]
    C = np.array(self.train['c'])
    R = np.diag([float(y*(1-y)) for y in Y])
    z = (self.X @ w)[:,0] - np.linalg.solve(R,(Y-C))
    lhs  = self.X.T @ R @ self.X
    rhs =  self.X.T @ R @ z
    w_new = np.linalg.solve(lhs, rhs)
    return w_new

  #Commmon Inteface
  def estimateParameters(self):
    if (not self.parametersEstimated):
      self.wList.append(np.array([0,0,0]))
      for i in range(1,self.iterations):
        self.wList.append(self.iterate(self.wList[i-1]))

      self.param  = self.wList[self.iterations - 1]
    self.parametersEstimated = True
    return self.param

  def computeBoundary(self):
    self.estimateParameters()
    w = self.param
    x1 = [(-w[2]*x2 - w[0])/w[1] for x2 in self.test['y']]
    return x1

  def computeMisclassificationRate(self):
   w = self.estimateParameters()
   for index, row in self.test.iterrows():
     x_with_bias = np.hstack([1, row['X']])
     pC_1lx = sigmoid(x_with_bias @ w)
     self.test.loc[index, 'GOOD']  = (pC_1lx > 0.5) and row['c'] == 1 or (pC_1lx <= 0.5)
   rate = ((self.test.shape[0] - sum(self.test['GOOD']))/self.test.shape[0])
   return rate

class LinearRegression():
  def __init__(self,train,test):
    self.train = train
    self.test = test
    self.parametersEstimated = False
    self.param = 0
    X = np.stack(self.train['X'])
    a = np.ones(len(X))
    a.shape = (len(a),1)
    self.X = np.hstack((a,X))

  def estimateParameters(self):
    if(not self.parametersEstimated):
      pseudoInverse = np.linalg.solve(self.X.T @ self.X, self.X.T)
      C = np.array(self.train['c'])
      self.param = pseudoInverse @ C
      self.parametersEstimated = True
    return self.param

  def computeBoundary(self):
    w = self.param
    x1 = [(0.5 - w[2]*x2 - w[0])/w[1] for x2 in self.test['y']]
    return x1
  def computeMisclassificationRate(self):
    w = self.estimateParameters()
    print(w)
    for index, row in self.test.iterrows():
      x_with_bias = np.hstack([1, row['X']])
      pC_1lx = np.inner(w, x_with_bias)
      self.test.loc[index, 'GOOD']  = (pC_1lx > 0.5) and row['c'] == 1 or (pC_1lx <= 0.5)
    rate = ((self.test.shape[0] - sum(self.test['GOOD']))/self.test.shape[0])
    return rate

class QDA():
  def __init__(self, train, test):
    self.train = train
    self.test = test
    self.parametersEstimated = False

  #Common Interface
  def estimateParameters(self):
    def estHyper(self):
      self.N1 = sum(self.train['c'])
      self.N2 = sum(1 - self.train['c'])
      self.N = self.N1 + self.N2
      self.pi1 = self.N1/self.N
      self.pi2 = self.N2/self.N
    def estMean(self):
      self.mu1 = sum([c*np.array(X) for c,X in \
                      zip(self.train['c'], self.train['X'])])/self.N1
      self.mu2 = sum([(1-c)*np.array(X) for c,X in \
                      zip(self.train['c'], self.train['X'])])/self.N2
    def estVariance(self):
      X = self.train['X']
      C = self.train['c']
      sumPart_1 = sum([c * np.outer(x - self.mu1, x - self.mu1) for x,c  in zip(X,C)])
      self.Sigma_1 = sumPart_1/self.N1
      sumPart_2 = sum([(1-c) * np.outer(x - self.mu2, x - self.mu2) for x,c in zip(X,C)])
      self.Sigma_2 = sumPart_2/self.N2
    if(not self.parametersEstimated):
      estHyper(self)
      estMean(self)
      estVariance(self)
      self.parametersEstimated = True
  def computeMisclassificationRate(self):
    self.estimateParameters()
    detRatio = 1/2 * np.log(np.linalg.det(self.Sigma_2)/ \
                            np.linalg.det(self.Sigma_1))
    bernouilliRatio = np.log(self.pi1/self.pi2)
    inv1 = np.linalg.inv(self.Sigma_1)
    inv2 = np.linalg.inv(self.Sigma_2)
    quadMatrix = inv1 - inv2
    lineMatrix = 1/2*(self.mu1.T @ inv1 - self.mu2.T @ inv2)
    for index, row in self.test.iterrows():
      x = np.array(row['X'])
      x.shape = (2,1)
      pC_1lx = -1/2*(x.T @ quadMatrix @ x) + x.T @ lineMatrix \
      + detRatio + bernouilliRatio
      self.test.loc[index, 'GOOD']  = (pC_1lx[0] > 0) and row['c'] == 1 or\
          (pC_1lx[0] <= 0)
    rate = ((self.test.shape[0] - sum(self.test['GOOD']))/self.test.shape[0])
    return rate

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

def sigmoid(z):
  return 1 / (1 + np.exp(-z))

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
        return directory
    except OSError:
        print ('Error: Creating directory. ' +  directory)

if __name__ == '__main__':
  main()

