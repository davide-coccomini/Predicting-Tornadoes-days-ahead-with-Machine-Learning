import numpy
import random
import os

from sklearn.model_selection import train_test_split
from collections import Counter

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

from sklearn.linear_model import Ridge
from sklearn.model_selection import validation_curve
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
import rotation_forest

import joblib

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
TORNADO_PATH = "../data/featuresExtracted/normal/quadrants9/tornado/"
OTHERWEATHER_PATH = "../data/featuresExtracted/normal/quadrants9/otherweather/"
FILE_NAME = "quadrants_day"
RESULT_FILE_NAME = "DAY4_9Q"

'''
# 4 QUADRANTS
TORNADO_PATH = "../data/featuresExtracted/normal/quadrants4/tornado/"
OTHERWEATHER_PATH = "../data/featuresExtracted/normal/quadrants4/otherweather/"
FILE_NAME = "quadrants_day"
RESULT_FILE_NAME = "DAY12345_4Q"

'''
# 4 QUADRANTS DISCRETIZED
TORNADO_PATH = "../data/dataDiscretized/quadrants4/tornado/"
OTHERWEATHER_PATH = "../data/dataDiscretized/quadrants4/otherweather/"
FILE_NAME = "quadrants_day"
RESULT_FILE_NAME = "DAY12345_4QD"

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

TRAINING_SET_PERCENTAGE = 95

def custom_train_test_split(dataset_values, dataset_labels, test_samples, skip_last_samples=0):
	#test_samples = int(((len(dataset_values) * (100-TRAINING_SET_PERCENTAGE))/100)/2) 
	if skip_last_samples != 0:
		return dataset_values[:-test_samples], dataset_labels[:-test_samples], dataset_values[-test_samples:-skip_last_samples], dataset_labels[-test_samples:-skip_last_samples]
	else:
		return dataset_values[:-test_samples], dataset_labels[:-test_samples], dataset_values[-test_samples:], dataset_labels[-test_samples:]


def main():
	# Read features
	#dataset_tornado = numpy.loadtxt(open(TORNADO_PATH + FILE_NAME, "r"), delimiter=",", skiprows=1)
	#dataset_otherweather = numpy.loadtxt(open(OTHERWEATHER_PATH + FILE_NAME, "r"), delimiter=",", skiprows=1)
	output_file = open("output.txt", "w+")

	for base_day in range(0, 5):
		print(str(base_day)+"/4")
		merged_dataset_tornado = []
		merged_dataset_otherweather = []
		for day in range(base_day, 5):
			dataset_tornado = numpy.loadtxt(open(TORNADO_PATH + FILE_NAME + str(day) + ".csv", "r"), delimiter=",", skiprows=1)
			dataset_otherweather = numpy.loadtxt(open(OTHERWEATHER_PATH + FILE_NAME + str(day) + ".csv", "r"), delimiter=",", skiprows=1)
			
			if(len(merged_dataset_tornado) == 0):
				merged_dataset_tornado = dataset_tornado
				merged_dataset_otherweather = dataset_otherweather
			else:
				merged_dataset_tornado = numpy.concatenate((numpy.array(merged_dataset_tornado), dataset_tornado), axis=1)
				merged_dataset_otherweather = numpy.concatenate(
					(numpy.array(merged_dataset_otherweather), dataset_otherweather), axis=1)

		dataset_tornado = merged_dataset_tornado
		dataset_otherweather = merged_dataset_otherweather
		print("_________________________________________________________________", file=output_file)
		print("CONSIDERED DAYS: ", 5 - base_day,  file=output_file)
		print("_________________________________________________________________",  file=output_file)
		print("___DATA SHAPES__",  file=output_file)
		print("Tornados:", dataset_tornado.shape, file=output_file)
		print("Others:", dataset_otherweather.shape, file=output_file)

		dataset_tornado = dataset_tornado[:, :321]
		dataset_otherweather = dataset_otherweather[:, :321]
		# Labelling
		tornado_labels = [[1] for i in range(0, len(dataset_tornado))]
		otherweather_labels = [[0] for i in range(0, len(dataset_otherweather))]

		dataset_tornado = numpy.append(dataset_tornado, tornado_labels, axis=1)
		dataset_otherweather = numpy.append(dataset_otherweather, otherweather_labels, axis=1)

		# Shuffle results
		#numpy.random.shuffle(complete_dataset)

		# Split in values and labels
		#dataset_values = complete_dataset[:, :-1]
		#dataset_labels = (complete_dataset[:, -1]).astype(int)

		# Split in test set and training set
		training_set_values_tornado, training_set_labels_tornado, test_set_values_tornado, test_set_labels_tornado = custom_train_test_split(
			dataset_tornado[:, :-1], (dataset_tornado[:, -1]).astype(int), 155, 45)

		training_set_values_otherweather, training_set_labels_otherweather, test_set_values_otherweather, test_set_labels_otherweather = custom_train_test_split(
			dataset_otherweather[:, :-1], (dataset_otherweather[:, -1]).astype(int), 225, 135)

		training_set_values = numpy.append(training_set_values_tornado, training_set_values_otherweather, axis=0)
		training_set_labels = numpy.append(training_set_labels_tornado, training_set_labels_otherweather, axis=0)
		test_set_values = numpy.append(test_set_values_tornado, test_set_values_otherweather, axis=0)
		test_set_labels = numpy.append(test_set_labels_tornado, test_set_labels_otherweather, axis=0)

		print(Counter(test_set_labels), file=output_file)
		weights = {0: 0.45, 1:0.55}
		# Try all the classifiers and select the best one
		classifiers = [
			KNeighborsClassifier(n_neighbors=1),
			KNeighborsClassifier(n_neighbors=2),
			KNeighborsClassifier(n_neighbors=3),
			KNeighborsClassifier(n_neighbors=5),
			KNeighborsClassifier(n_neighbors=8),
			KNeighborsClassifier(n_neighbors=12),
			KNeighborsClassifier(n_neighbors=16),
			DecisionTreeClassifier(class_weight=weights),
			RandomForestClassifier(class_weight=weights),
			GaussianNB(),
			SVC(kernel="linear", class_weight=weights),
			SVC(kernel="linear", C=0.025, class_weight=weights),
			AdaBoostClassifier(),
			MLPClassifier(alpha=1, max_iter=10000)
		]
		best_classifier = classifiers[0]
		best_score = 0
		for classifier in classifiers:
			print("_________________", file=output_file)
			print(classifier, file=output_file)
			print(len(training_set_values), len(training_set_labels), file=output_file)
			clf = classifier.fit(training_set_values, training_set_labels)
			predicted = clf.predict(test_set_values)
			score = round(f1_score(test_set_labels, predicted),2)
			precision = round(precision_score(test_set_labels, predicted),2)
			accuracy = round(accuracy_score(test_set_labels, predicted),2)
			recall = round(recall_score(test_set_labels, predicted),2)

			print("Confusion matrix:", file=output_file)
			matrix = confusion_matrix(test_set_labels, predicted)
			FP = matrix.sum(axis=0) - numpy.diag(matrix)  
			FN = matrix.sum(axis=1) - numpy.diag(matrix)
			TP = numpy.diag(matrix)
			TN = matrix.sum() - (FP + FN + TP)

			FP = FP.astype(float)
			FN = FN.astype(float)
			TP = TP.astype(float)
			TN = TN.astype(float)
			# Sensitivity, hit rate, recall, or true positive rate
			TPR = TP/(TP+FN)
			# Specificity or true negative rate
			TNR = TN/(TN+FP) 
			far = round(matrix[0][1]/(matrix[1][1]+matrix[0][0]),2)
			#csi = round(matrix[1][1]/(matrix[1][1]+matrix[0][1]+matrix[1][0]),2)
			print("F1-Score: " + str(score), "Precision: ", precision, "Recall: ", recall, "Accuracy:", accuracy, "far:", far, "TPR:", TPR, "TNR:", TNR, file=output_file)
			print(matrix, file=output_file)
			if score > best_score:
				best_score = score
				best_classifier = clf

			'''
			scores = cross_val_score(clf, dataset_values, dataset_labels, cv=5)
			print("Cross validation:")
			print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
			
			if scores.mean() > best_accuracy:
				best_accuracy = scores.mean()
				best_classifier = clf
			'''

		print("\nSelected classifier:", file=output_file)
		print(best_classifier, file=output_file)
		
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

	output_file.close()

main()
