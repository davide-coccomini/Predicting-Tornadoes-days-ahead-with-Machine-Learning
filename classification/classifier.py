import numpy
import random
import os

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.linear_model import Ridge
from sklearn.model_selection import validation_curve
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve


from sklearn.externals import joblib

'''
# 9 QUADRANTS SELECTED
TORNADO_PATH = "../data/featuresSelected/quadrants9/tornado/"
OTHERWEATHER_PATH = "../data/featuresSelected/quadrants9/otherweather/"
FILE_NAME = "quadrants_day4.csv"
RESULT_FILE_NAME = "DAY4_9QS160"


# 4 QUADRANTS SELECTED
TORNADO_PATH = "../data/featuresSelected/quadrants4/tornado/"
OTHERWEATHER_PATH = "../data/featuresSelected/quadrants4/otherweather/"
FILE_NAME = "quadrants_day4.csv"
RESULT_FILE_NAME = "DAY4_9QS60"


# 9 QUADRANTS 
TORNADO_PATH = "../data/featuresExtracted/quadrants9/tornado/"
OTHERWEATHER_PATH = "../data/featuresExtracted/quadrants9/otherweather/"
FILE_NAME = "quadrants_day4.csv"
RESULT_FILE_NAME = "DAY4_9Q"

# 4 QUADRANTS
TORNADO_PATH = "../data/featuresExtracted/quadrants4/tornado/"
OTHERWEATHER_PATH = "../data/featuresExtracted/quadrants4/otherweather/"
FILE_NAME = "quadrants_day4.csv"
RESULT_FILE_NAME = "DAY4_4Q"
'''
# 4 QUADRANTS DISCRETIZED
TORNADO_PATH = "../data/dataDiscretized/quadrants4/tornado/"
OTHERWEATHER_PATH = "../data/dataDiscretized/quadrants4/otherweather/"
FILE_NAME = "quadrants_day"
RESULT_FILE_NAME = "DAY12345_4QD"
'''
# 9 QUADRANTS DISCRETIZED
TORNADO_PATH = "../data/dataDiscretized/quadrants9/tornado/"
OTHERWEATHER_PATH = "../data/dataDiscretized/quadrants9/otherweather/"
FILE_NAME = "quadrants_day4.csv"
RESULT_FILE_NAME = "DAY4_9QD"

# 4 QUADRANTS DISCRETIZED AND SELECTED
TORNADO_PATH = "../data/dataDiscretized/quadrants4/tornado/"
OTHERWEATHER_PATH = "../data/dataDiscretized/quadrants4/otherweather/"
FILE_NAME = "quadrants_day4.csv"
RESULT_FILE_NAME = "DAY4_4QDS60"

# 9 QUADRANTS DISCRETIZED AND SELECTED
TORNADO_PATH = "../data/dataDiscretized/quadrants9/tornado/"
OTHERWEATHER_PATH = "../data/dataDiscretized/quadrants9/otherweather/"
FILE_NAME = "quadrants_day4.csv"
RESULT_FILE_NAME = "DAY4_9QDS160"

# RAW EXTENDED
TORNADO_PATH = "../data/collections/csv/normal/tornado/"
OTHERWEATHER_PATH = "../data/collections/csv/normal/otherweather/"
FILE_NAME = "day4.csv"
RESULT_FILE_NAME = "DAY4_RE"

# Features extracted from extended 4 quadrants
TORNADO_PATH = "../data/featuresExtracted/extended/quadrants4/tornado/"
OTHERWEATHER_PATH = "../data/featuresExtracted/extended/quadrants4/otherweather/"
FILE_NAME = "quadrants_day4.csv"
RESULT_FILE_NAME = "DAY4_RE_4Q"

# Features extracted from extended 9 quadrants
TORNADO_PATH = "../data/featuresExtracted/extended/quadrants9/tornado/"
OTHERWEATHER_PATH = "../data/featuresExtracted/extended/quadrants9/otherweather/"
FILE_NAME = "quadrants_day4.csv"
RESULT_FILE_NAME = "DAY4_RE_9Q"

# Days evolution 4 quadrants
TORNADO_PATH = "../data/daysEvolution/quadrants4/tornado.csv"
OTHERWEATHER_PATH = "../data/daysEvolution//quadrants4/otherweather.csv"
FILE_NAME = ""
RESULT_FILE_NAME = "DAYS_EV_4Q"

# Days evolution 9 quadrants
TORNADO_PATH = "../data/daysEvolution/quadrants9/tornado.csv"
OTHERWEATHER_PATH = "../data/daysEvolution//quadrants9/otherweather.csv"
FILE_NAME = ""
RESULT_FILE_NAME = "DAYS_EV_9Q"

# Days evolution 4 quadrants discretized
TORNADO_PATH = "../data/dataDiscretized/quadrants4Evolution/tornado/tornado.csv"
OTHERWEATHER_PATH = "../data/dataDiscretized//quadrants4Evolution/otherweather/otherweather.csv"
FILE_NAME = ""
RESULT_FILE_NAME = "DAYS_EV_4QD"

# Days evolution 9 quadrants discretized
TORNADO_PATH = "../data/dataDiscretized/quadrants9Evolution/tornado/tornado.csv"
OTHERWEATHER_PATH = "../data/dataDiscretized/quadrants9Evolution/otherweather/otherweather.csv"
FILE_NAME = ""
RESULT_FILE_NAME = "DAYS_EV_9QD"
'''

TRAINING_SET_PERCENTAGE = 85


def main():
	# Read features
	dataset_tornado = numpy.loadtxt(open(TORNADO_PATH + FILE_NAME, "r"), delimiter=",", skiprows=1)
	dataset_otherweather = numpy.loadtxt(open(OTHERWEATHER_PATH + FILE_NAME, "r"), delimiter=",", skiprows=1)

	merged_dataset_tornado = []
	merged_dataset_otherweather = []
	for day in range(0, 5):
		dataset_tornado = numpy.loadtxt(open(TORNADO_PATH + FILE_NAME + str(day) + ".csv", "r"), delimiter=",", skiprows=1)
		dataset_otherweather = numpy.loadtxt(open(OTHERWEATHER_PATH + FILE_NAME + str(day) + ".csv", "r"), delimiter=",", skiprows=1)
		print(dataset_tornado.shape)
		print(dataset_otherweather.shape)
		if(len(merged_dataset_tornado) == 0):
			merged_dataset_tornado = dataset_tornado
			merged_dataset_otherweather = dataset_otherweather
		else:
			merged_dataset_tornado = numpy.concatenate((numpy.array(merged_dataset_tornado), dataset_tornado), axis=1)
			merged_dataset_otherweather = numpy.concatenate(
				(numpy.array(merged_dataset_otherweather), dataset_otherweather), axis=1)

	dataset_tornado = merged_dataset_tornado
	dataset_otherweather = merged_dataset_otherweather


	dataset_tornado = dataset_tornado[:, :321]
	dataset_otherweather = dataset_otherweather[:, :321]
	# Labelling
	tornado_labels = [[1] for i in range(0, len(dataset_tornado))]
	otherweather_labels = [[0] for i in range(0, len(dataset_otherweather))]

	dataset_tornado = numpy.append(dataset_tornado, tornado_labels, axis=1)
	dataset_otherweather = numpy.append(dataset_otherweather, otherweather_labels, axis=1)
	complete_dataset = numpy.append(dataset_tornado, dataset_otherweather, axis=0)

	# Shuffle results
	numpy.random.shuffle(complete_dataset)

	# Split in values and labels
	dataset_values = complete_dataset[:, :-1]
	dataset_labels = (complete_dataset[:, -1]).astype(int)

	# Split in test set and training set
	training_set_values, test_set_values, training_set_labels, test_set_labels = train_test_split(dataset_values, dataset_labels, test_size=(100 - TRAINING_SET_PERCENTAGE) / 100, random_state=42)

	# Try all the classifiers and select the best one
	classifiers = [
		KNeighborsClassifier(1),
		SVC(kernel="linear", C=0.025),
		DecisionTreeClassifier(),
		RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
		GaussianNB()
	]

	best_classifier = classifiers[0]
	best_accuracy = 0
	for classifier in classifiers:
		clf = classifier.fit(training_set_values, training_set_labels)
		predicted = clf.predict(test_set_values)
		score = precision_score(test_set_labels, predicted, average='weighted')

		print("Accuracy: " + str(score))
		print("Confusion matrix:")
		print(confusion_matrix(test_set_labels, predicted))
		scores = cross_val_score(clf, dataset_values, dataset_labels, cv=5)
		print("Cross validation:")
		print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

		if scores.mean() > best_accuracy:
			best_accuracy = scores.mean()
			best_classifier = clf

	print("\nSelected classifier:")
	print(best_classifier)

	# Save the trained classifier
	result_folder_name = "trainedClassifiers/classic/" + str(int(round(score * 100))) + "_" + RESULT_FILE_NAME
	if not os.path.isdir(result_folder_name):
		os.mkdir(result_folder_name)

	training_set_values_file = open(result_folder_name + "/training_set_values.csv", "w")
	test_set_values_file = open(result_folder_name + "/test_set_values.csv", "w")
	training_set_labels_file = open(result_folder_name + "/training_set_labels.csv", "w")
	test_set_labels_file = open(result_folder_name + "/test_set_labels.csv", "w")

	numpy.savetxt(training_set_values_file, training_set_values, delimiter=",", comments="")
	numpy.savetxt(test_set_values_file, test_set_values, delimiter=",", comments="")
	numpy.savetxt(training_set_labels_file, training_set_labels, delimiter=",", comments="")
	numpy.savetxt(test_set_labels_file, test_set_labels, delimiter=",", comments="")

	filename = result_folder_name + "/classifier.sav"
	joblib.dump(best_classifier, filename)


main()
