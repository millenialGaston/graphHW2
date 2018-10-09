# HW2 Frederic Boileau
The code for the homework is in the folder hw2Py and can be 
imported as a module or run as a script. Detailed explanations
of its structure and its classes are included in the latex-generated
pdf which also contains the theoretical results.

For the script to run properly the module must be in a repository
which also contains the folder hwk2data which contains the text files
with the same names as when given by Pr. Lacoste.

If run as a script it will plot the test data as a scatter plot,
one for each dataset (i.e. A, B and C) with the four decision boundaries 
included.

There is a class for each classification model/process which all
share a common interface: estimateParameters, computeBoundary and
computeMisclassificationRate.

All those classes and methods use the class Data which
fetches and parses the data when initialized with the 
letter of the corresponding data set and contains
two panda structures, one for the training set and
one for the testing set, named train and test respectively.
