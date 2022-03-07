import os
import numpy
from statistics import mode

from sklearn.externals import joblib

from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score


#TRAINED_CLASSIFIER_FOLDER = "../classification/trainedClassifiers/classic/74_DAY4_4Q/"
TRAINED_CLASSIFIER_FOLDER = "../classification/trainedClassifiers/ensemble/81_4Q/"
CLASSIFIER_TYPE = 1   # 0: CLASSIC; 1: ENSEMBLE
CONSIDERED_DAYS = [0, 5] # Only for ensemble classifiers
NUM_FEATURES_PER_DAY = 80  # Only for ensemble classifiers
def main():
	if CLASSIFIER_TYPE == 0:
		print(CLASSIFIER_TYPE)
		loaded_classifier = joblib.load(TRAINED_CLASSIFIER_FOLDER + "classifier.sav")
		test_set_values = numpy.loadtxt(open(TRAINED_CLASSIFIER_FOLDER + "test_set_values.csv", "r"), delimiter=",")
		test_set_labels = numpy.loadtxt(open(TRAINED_CLASSIFIER_FOLDER + "test_set_labels.csv", "r"), delimiter=",")
		accuracy = loaded_classifier.score(test_set_values, test_set_labels)
		print("Accuracy:"+str(accuracy))
	if CLASSIFIER_TYPE == 1:

		print(CLASSIFIER_TYPE)
		test_set_values = numpy.loadtxt(open(TRAINED_CLASSIFIER_FOLDER + "test_set_values.csv", "r"), delimiter=",")
		test_set_labels = numpy.loadtxt(open(TRAINED_CLASSIFIER_FOLDER + "test_set_labels.csv", "r"), delimiter=",")

		classifiers = []
		for day in range(CONSIDERED_DAYS[0], CONSIDERED_DAYS[1]):
			loaded_classifier = joblib.load(TRAINED_CLASSIFIER_FOLDER + "day"+str(day)+".sav")
			classifiers.append(loaded_classifier)
		
		predictions = []

		for day in range(CONSIDERED_DAYS[0], CONSIDERED_DAYS[1]):
			min_column = day * NUM_FEATURES_PER_DAY
			max_column = ((day + 1) * NUM_FEATURES_PER_DAY)
			predicted = classifiers[day].predict(test_set_values[:, min_column:max_column])
			predictions.append(predicted)
		predictions = numpy.array(predictions)

		# Collapse each of 5 rows in one value, the most choosen one
		ensemble_predictions = []
		for i in range(0, len(predictions[0])):
			most_choosen = mode(predictions[:, i])
			ensemble_predictions.append(most_choosen)
		ensemble_predictions = numpy.array(ensemble_predictions)

		# Compare the ensemble predictions with right ones
		score = precision_score(test_set_labels, ensemble_predictions, average='weighted')
		print("Accuracy: " + str(score))
		print("Confusion matrix:")
		print(confusion_matrix(test_set_labels, predicted))
main()
