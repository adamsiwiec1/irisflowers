# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Tutorial link:
# https://machinelearningmastery.com/machine-learning-in-python-step-by-step/

# ***Part 1 - Import data, take a peek, and create basic plots.*** #

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# take a peek at your data - show the first 30 lines
print(dataset.head(20))

# descriptions - shows details of the dataset
print(dataset.describe())

# class distribution
print(dataset.groupby('class').size())

# Data Visualization

# We will use two types of plots:
# univariate - plot to better understand each attribute
# multivariate - plot to better understand relationships between attributes

# Univariate plot examples #

# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()

# histograms
dataset.hist()
pyplot.show()

# *** Multivariate plot examples *** #

# Note the diagonal grouping of some pairs of attributes.
# This suggests a high correlation and a predictable relationship.

# scatter plot matrix
scatter_matrix(dataset)
pyplot.show()


# *** Part 2 - Create algorithms *** #


# Part 2.1 - Create a Validation Dataset

# To test if our model is good, we need a validation dataset. To do this, we
# will use 80% of our data to train and hold back the other 20% for validation.

# Split-out validation dataset
array = dataset.values
X = array[:,0:4] # Sepal-length, sepal-width, petal-length, petal-width
y = array[:,4]

# Our two datasets
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

# Part 2.2 - Test Harness

# We will use stratified 10-fold cross validation to estimate model accuracy.

# This will split our dataset into 10 parts, train on 9 and test on 1 and repeat
# for all combinations of train-test splits.

# Stratified means that each fold or split of the dataset will aim to have the
# same distribution of example by class as exist in the whole training dataset.

# We are using the metric of ‘accuracy‘ to evaluate models.

# This is a ratio of the number of correctly predicted instances.

# We will be using the scoring variable when we run build and evaluate each model next.

# Part 2.3 - Build Models

# We don’t know which algorithms would be good on this problem or what configurations to use.

# We get an idea from the plots that some of the classes are partially linearly separable in
# some dimensions, so we are expecting generally good results.

# We are testing 6 algoritms:
# Logistic Regression (LR)
# Linear Discriminant Analysis (LDA)
# K-Nearest Neighbors (KNN).
# Classification and Regression Trees (CART).
# Gaussian Naive Bayes (NB).
# Support Vector Machines (SVM).

# This is a good mixture of simple linear (LR and LDA), nonlinear (KNN, CART, NB and SVM) algorithms.

# Time to build and evaluate our models:

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Running the program after adding this code above will show us the accuracy score for each model:
# LR: 0.941667 (0.065085)
# LDA: 0.975000 (0.038188)
# KNN: 0.958333 (0.041667)
# CART: 0.950000 (0.040825)
# NB: 0.950000 (0.055277)
# SVM: 0.983333 (0.033333)

# Part 2.3 - Build Models
# *** Results may vary each run ***
# Consider running the example a few times and compare the average outcomes.

# Support Vector Machines (SVM) has the largest estimated accuracy score at about 0.98 or 98%

# A useful way to compare the samples of results for each algorithm is to create a box and whisker
# plot for each distribution and compare the distributions. Shown below.

# Compare Algorithms
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()

# Part 3 - Make predictions

# We must choose an algorithm to predict with, in our case we
# choose SVM bcuz it had the highest accuracy score above.

# Make predictions on validation dataset
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# Part 3.1 - Evaluate predictions

print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))