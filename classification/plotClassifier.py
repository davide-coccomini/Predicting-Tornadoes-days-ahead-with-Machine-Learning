import warnings
import numpy as np
import pandas as pd
from pylab import rcParams
import matplotlib.pyplot as plt
from sklearn import neighbors
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
# filter warnings
warnings.filterwarnings("ignore")


TRAINED_CLASSIFIER_FOLDER = "../classification/trainedClassifiers/classic/83_DAYS_EV_4Q/"


def classify_and_plot():
	''' 
	split data, fit, classify, plot and evaluate results 
	
	# split data into training and testing set
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.33, random_state=41)
	'''

	X_train = np.loadtxt(open(TRAINED_CLASSIFIER_FOLDER +
                           "training_set_values.csv", "r"), delimiter=",")[:, :2]
	y_train = np.loadtxt(open(TRAINED_CLASSIFIER_FOLDER +
                           "training_set_labels.csv", "r"), delimiter=",")[:]
	X_test = np.loadtxt(open(TRAINED_CLASSIFIER_FOLDER +
                          "test_set_values.csv", "r"), delimiter=",")[:, :2]
	y_test = np.loadtxt(open(TRAINED_CLASSIFIER_FOLDER +
                          "test_set_labels.csv", "r"), delimiter=",")[:]

	X = np.concatenate((X_train, X_test), axis=0)
	y = np.concatenate((y_train, y_test), axis=0)

	# init vars
	n_neighbors = 1
	h = .02  # step size in the mesh

	# Create color maps
	cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
	cmap_bold = ListedColormap(['#FF0000', '#0000FF'])

	rcParams['figure.figsize'] = 5, 5
	for weights in ['uniform', 'distance']:
		# we create an instance of Neighbours Classifier and fit the data.
		clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
		clf.fit(X_train, y_train)
		# Plot the decision boundary. For that, we will assign a color to each
		# point in the mesh [x_min, x_max]x[y_min, y_max].
		x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
		y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
		xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                       np.arange(y_min, y_max, h))
		Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
		# Put the result into a color plot
		Z = Z.reshape(xx.shape)
		fig = plt.figure()
		plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

		# Plot also the training points, x-axis = 'Glucose', y-axis = "BMI"
		plt.scatter(X[:100, 0], X[:100, 1], c=y[:100],
		            cmap=cmap_bold, edgecolor='k', s=20)

		plt.xlim(xx.min(), xx.max())
		plt.ylim(yy.min(), yy.max())
		plt.title("0/1 outcome classification (k = %i, weights = '%s') [100 SAMPLES]" %
                    (n_neighbors, weights))
		plt.show()

		fig = plt.figure()
		plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
		plt.xlim(xx.min(), xx.max())
		plt.ylim(yy.min(), yy.max())
		plt.scatter(X[:, 0], X[:, 1], c=y[:], cmap=cmap_bold, edgecolor='k', s=20)
		plt.title("0/1 outcome classification (k = %i, weights = '%s') [ALL SAMPLES]" %
                    (n_neighbors, weights))
		plt.show()
		fig.savefig(weights + '.png')

		# evaluate
		y_expected = y_test
		y_predicted = clf.predict(X_test)


# classify, evaluate and plot results
classify_and_plot()
