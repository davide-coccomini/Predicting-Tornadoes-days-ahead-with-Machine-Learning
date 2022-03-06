import numpy
import random
import os

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

from statistics import mode

from sklearn.externals import joblib

CONSIDERED_DAYS = [0, 5]
'''
# Selected 4 quadrants
TORNADO_PATH = "../data/featuresSelected/quadrants4/tornado/"
OTHERWEATHER_PATH = "../data/featuresSelected/quadrants4/otherweather/"
FILE_NAME = "quadrants_day"
NUM_FEATURES_PER_DAY = 70
RESULT_FILE_NAME = "4QS" + str(NUM_FEATURES_PER_DAY)

# Selected 9 quadrants
TORNADO_PATH = "../data/featuresSelected/quadrants9/tornado/"
OTHERWEATHER_PATH = "../data/featuresSelected/quadrants9/otherweather/"
FILE_NAME = "quadrants_day"
NUM_FEATURES_PER_DAY = 140
RESULT_FILE_NAME = "9QS" + str(NUM_FEATURES_PER_DAY)

# Extracted 4 quadrants
TORNADO_PATH = "../data/featuresExtracted/quadrants4/tornado/"
OTHERWEATHER_PATH = "../data/featuresExtracted/quadrants4/otherweather/"
FILE_NAME = "quadrants_day"
NUM_FEATURES_PER_DAY = 80
RESULT_FILE_NAME = "4Q"

# Extracted 9 quadrants
TORNADO_PATH = "../data/featuresExtracted/quadrants9/tornado/"
OTHERWEATHER_PATH = "../data/featuresExtracted/quadrants9/otherweather/"
FILE_NAME = "quadrants_day"
NUM_FEATURES_PER_DAY = 180  # Num_quadrants(9) * Num_variables(10) * Num_Features(mean and std -> 2)
RESULT_FILE_NAME = "9Q"

# Raw collections {impossible for memory allocation}
TORNADO_PATH = "../data/collections/csv/tornado/"
OTHERWEATHER_PATH = "../data/collections/csv/otherweather/"
FILE_NAME = "day"
NUM_FEATURES_PER_DAY = 3610
RESULT_FILE_NAME = "RAW"


# Extracted 9 quadrants discretized
TORNADO_PATH = "../data/dataDiscretized/quadrants9/tornado/"
OTHERWEATHER_PATH = "../data/dataDiscretized/quadrants9/otherweather/"
FILE_NAME = "quadrants_day"
NUM_FEATURES_PER_DAY = 180  # Num_quadrants(9) * Num_variables(10) * Num_Features(mean and std -> 2)
RESULT_FILE_NAME = "9QD"

# Extracted 4 quadrants discretized
TORNADO_PATH = "../data/dataDiscretized/quadrants4/tornado/"
OTHERWEATHER_PATH = "../data/dataDiscretized/quadrants4/otherweather/"
FILE_NAME = "quadrants_day"
NUM_FEATURES_PER_DAY = 80 
RESULT_FILE_NAME = "4QD" + str(CONSIDERED_DAYS[1])

# Selected 4 quadrants discretized
TORNADO_PATH = "../data/dataDiscretized/quadrants4Selected/tornado/"
OTHERWEATHER_PATH = "../data/dataDiscretized/quadrants4Selected/otherweather/"
FILE_NAME = "quadrants_day"
NUM_FEATURES_PER_DAY = 70
RESULT_FILE_NAME = "4QDS" + str(NUM_FEATURES_PER_DAY)

# Selected 9 quadrants discretized
TORNADO_PATH = "../data/dataDiscretized/quadrants9Selected/tornado/"
OTHERWEATHER_PATH = "../data/dataDiscretized/quadrants9Selected/otherweather/"
FILE_NAME = "quadrants_day"
NUM_FEATURES_PER_DAY = 140
RESULT_FILE_NAME = "9QDS" + str(NUM_FEATURES_PER_DAY)

# Extracted 4 quadrants extended
TORNADO_PATH = "../data/featuresExtracted/extended/quadrants4/tornado/"
OTHERWEATHER_PATH = "../data/featuresExtracted/extended/quadrants4/otherweather/"
FILE_NAME = "quadrants_day"
NUM_FEATURES_PER_DAY = 80
RESULT_FILE_NAME = "EXT_4Q"

# Extracted 9 quadrants extended
TORNADO_PATH = "../data/featuresExtracted/extended/quadrants9/tornado/"
OTHERWEATHER_PATH = "../data/featuresExtracted/extended/quadrants9/otherweather/"
FILE_NAME = "quadrants_day"
NUM_FEATURES_PER_DAY = 180
RESULT_FILE_NAME = "EXT_9Q"
'''

TORNADO_PATH = "../data/featuresExtracted/normalized/quadrants4/tornado/"
OTHERWEATHER_PATH = "../data/featuresExtracted/normalized/quadrants4/otherweather/"
FILE_NAME = "quadrants_day"
NUM_FEATURES_PER_DAY = 80
RESULT_FILE_NAME = "4QN" + str(CONSIDERED_DAYS[1])

TRAINING_SET_PERCENTAGE = 80
NUMBER_ITERATIONS = 15


def main():
	# Merge all days in one matrix, each row will contain all days features horizontally ordered from 0 to 5
	merged_dataset_tornado = []
	merged_dataset_otherweather = []
	print("Reading datasets ...")
	for day in range(CONSIDERED_DAYS[0], CONSIDERED_DAYS[1]):
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

	print("Datasets readed.")

	# Labelling
	tornado_labels = [[1] for i in range(0, len(merged_dataset_tornado))]
	otherweather_labels = [[0] for i in range(0, len(merged_dataset_otherweather))]
	dataset_tornado = numpy.append(merged_dataset_tornado, tornado_labels, axis=1)
	dataset_otherweather = numpy.append(merged_dataset_otherweather, otherweather_labels, axis=1)
	complete_dataset = numpy.append(dataset_tornado, dataset_otherweather, axis=0)

	# Train one classifier per day
	print("Training classifiers...")
	trained_classifiers = None
	best_average_score = 0
	selected_training_set_values = None
	selected_training_set_labels = None
	selected_test_set_values = None
	selected_test_set_labels = None
	for iteration in range(0, NUMBER_ITERATIONS):  # Try to train many classifiers to achieve the best accuracy
		# Shuffle results
		numpy.random.shuffle(complete_dataset)

		# Split in values and labels
		dataset_values = complete_dataset[:, :-1]
		dataset_labels = (complete_dataset[:, -1]).astype(int)

		# Split in test set and training set
		training_set_values, test_set_values, training_set_labels, test_set_labels = train_test_split(
			dataset_values, dataset_labels, test_size=(100 - TRAINING_SET_PERCENTAGE) / 100, random_state=42, shuffle=True)
		total_scores = 0

		tmp_classifiers = []
		for day in range(CONSIDERED_DAYS[0], CONSIDERED_DAYS[1]):
			min_column = day * NUM_FEATURES_PER_DAY
			max_column = ((day + 1) * NUM_FEATURES_PER_DAY)
			clf = KNeighborsClassifier(1).fit(training_set_values[:, min_column:max_column], training_set_labels)
			predicted = clf.predict(test_set_values[:, min_column:max_column])
			score = precision_score(test_set_labels, predicted, average='weighted')
			total_scores += score
			tmp_classifiers.append(clf)
		average_score = total_scores / (CONSIDERED_DAYS[1])
		if(average_score > best_average_score):
			best_average_score = average_score
			trained_classifiers = tmp_classifiers
			selected_training_set_values = training_set_values
			selected_training_set_labels = training_set_labels
			selected_test_set_values = test_set_values
			selected_test_set_labels = test_set_labels
			if(iteration != NUMBER_ITERATIONS - 1):
				print("New best average accuracy achieved: " + str(best_average_score))

	print("Best accuracy achieved in " + str(NUMBER_ITERATIONS) + " iterations: " + str(best_average_score))

	# Test accuracy of ensemble classifier
	# Each classifier try to make a prediction based on the data of the tornado of its assigned day
	print("Testing ensemble classifier ...")
	predictions = []
	for day in range(CONSIDERED_DAYS[0], CONSIDERED_DAYS[1]):
		min_column = day * NUM_FEATURES_PER_DAY
		max_column = ((day + 1) * NUM_FEATURES_PER_DAY)
		predicted = trained_classifiers[day].predict(selected_test_set_values[:, min_column:max_column])
		predictions.append(predicted)
	predictions = numpy.array(predictions)


	# Collapse each of 5 rows in one value, the most choosen one
	ensemble_predictions = []
	for i in range(0, len(predictions[0])):
		most_choosen = mode(predictions[:, i])
		ensemble_predictions.append(most_choosen)
	ensemble_predictions = numpy.array(ensemble_predictions)

	# Compare the ensemble predictions with right ones
	score = precision_score(selected_test_set_labels, ensemble_predictions, average='weighted')
	print("Accuracy: " + str(score))
	print("Confusion matrix:")
	print(confusion_matrix(selected_test_set_labels, predicted))

	# Save the trained classifier
	result_folder_name = "trainedClassifiers/ensemble/" + str(int(round(score * 100))) + "_" + RESULT_FILE_NAME
	print(str(round(score * 100)))
	if not os.path.isdir(result_folder_name):
		os.mkdir(result_folder_name)

		for day in range(CONSIDERED_DAYS[0], CONSIDERED_DAYS[1]):
			classifier = trained_classifiers[day]
			filename = result_folder_name + "/day" + str(day) + ".sav"
			joblib.dump(classifier, filename)

		training_set_values_file = open(result_folder_name + "/training_set_values.csv", "w")
		test_set_values_file = open(result_folder_name + "/test_set_values.csv", "w")
		training_set_labels_file = open(result_folder_name + "/training_set_labels.csv", "w")
		test_set_labels_file = open(result_folder_name + "/test_set_labels.csv", "w")

		numpy.savetxt(training_set_values_file, selected_training_set_values, delimiter=",", comments="")
		numpy.savetxt(test_set_values_file, selected_test_set_values, delimiter=",", comments="")
		numpy.savetxt(training_set_labels_file, selected_training_set_labels, delimiter=",", comments="")
		numpy.savetxt(test_set_labels_file, selected_test_set_labels, delimiter=",", comments="")

	print(selected_training_set_values.shape)


main()
