import numpy
import random
import os
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import MinMaxScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix

'''
# 4 QUADRANTS
TORNADO_PATH = "../data/featuresExtracted/quadrants4/tornado/"
OTHERWEATHER_PATH = "../data/featuresExtracted/quadrants4/otherweather/"
FILE_NAME = "quadrants_day"
RESULT_DIRECTORY = "../data/dataDiscretized/quadrants4/"

# 9 QUADRANTS
TORNADO_PATH = "../data/featuresExtracted/quadrants9/tornado/"
OTHERWEATHER_PATH = "../data/featuresExtracted/quadrants9/otherweather/"
FILE_NAME = "quadrants_day"
RESULT_DIRECTORY = "../data/dataDiscretized/quadrants9/"


# 4 QUADRANTS SELECTED
TORNADO_PATH = "../data/featuresSelected/quadrants4/tornado/"
OTHERWEATHER_PATH = "../data/featuresSelected/quadrants4/otherweather/"
FILE_NAME = "quadrants_day"
RESULT_DIRECTORY = "../data/dataDiscretized/quadrants4Selected/"

# 9 QUADRANTS SELECTED
TORNADO_PATH = "../data/featuresSelected/quadrants9/tornado/"
OTHERWEATHER_PATH = "../data/featuresSelected/quadrants9/otherweather/"
FILE_NAME = "quadrants_day"
RESULT_DIRECTORY = "../data/dataDiscretized/quadrants9Selected/"

# Days evolution 4 quadrants
TORNADO_PATH = "../data/daysEvolution/quadrants4/tornado"
OTHERWEATHER_PATH = "../data/daysEvolution//quadrants4/otherweather"
FILE_NAME = ""
RESULT_DIRECTORY = "../data/dataDiscretized/quadrants4Evolution/"
WHOLE_FILE = True
'''
# Days evolution 9 quadrants
TORNADO_PATH = "../data/daysEvolution/quadrants9/tornado"
OTHERWEATHER_PATH = "../data/daysEvolution//quadrants9/otherweather"
FILE_NAME = ""
RESULT_DIRECTORY = "../data/dataDiscretized/quadrants9Evolution/"
WHOLE_FILE = True

TRAINING_SET_PERCENTAGE = 85



def main():

	for day in range(0, 6):
		if WHOLE_FILE: # Skip the day if there is no distinction in the files (days collapsed)
			day = ""
		# Read features
		dataset_tornado = numpy.loadtxt(open(TORNADO_PATH + FILE_NAME + str(day) + ".csv", "rb"), delimiter=",", skiprows=1)
		dataset_otherweather = numpy.loadtxt(open(OTHERWEATHER_PATH + FILE_NAME + str(day) + ".csv", "rb"), delimiter=",", skiprows=1)

		# Labelling
		tornado_labels = [[1] for i in range(0, len(dataset_tornado))]
		otherweather_labels = [[0] for i in range(0, len(dataset_otherweather))]

		dataset_tornado = numpy.append(dataset_tornado, tornado_labels, axis=1)
		dataset_otherweather = numpy.append(dataset_otherweather, otherweather_labels, axis=1)
		complete_dataset = numpy.append(dataset_tornado, dataset_otherweather,axis=0)

		# Shuffle results
		numpy.random.shuffle(complete_dataset)
		
		# Split in values and labels
		dataset_values = complete_dataset[:,:-1]
		dataset_labels = (complete_dataset[:,-1]).astype(int)
		
		# Split in test set and training set
		training_set_values, test_set_values, training_set_labels, test_set_labels = train_test_split(dataset_values, dataset_labels, test_size=(100-TRAINING_SET_PERCENTAGE)/100, random_state = 42)
	
		classifiers = [
			KNeighborsClassifier(1),
			SVC(kernel="linear", C=0.025), 
			DecisionTreeClassifier(),
			RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
			GaussianNB()
		]

		best_bins = []
		best_accuracy = 0
		best_classifier = classifiers[0]
		best_discretizer = None
		
		if not os.path.isdir(RESULT_DIRECTORY):
			os.mkdir(RESULT_DIRECTORY)
			os.mkdir(RESULT_DIRECTORY + "tornado")
			os.mkdir(RESULT_DIRECTORY + "otherweather")

		for s in range(0,20): # Make N trial
			bins = [int(random.randrange(400,1000)) for i in range(0, len(complete_dataset[0])-1)] # Fullfill randomly the number of bins for each feature
			discretizer = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy="uniform").fit(complete_dataset[:, :-1])
			discretized_training_set_values = discretizer.transform(training_set_values)
			discretized_test_set_values = discretizer.transform(test_set_values)
			
			for classifier in classifiers: # Try all the classifiers
				clf = classifier.fit(discretized_training_set_values,
										training_set_labels)
				predicted = clf.predict(discretized_test_set_values)
				score = precision_score(test_set_labels, predicted, average='weighted')

				
				if score > best_accuracy:
					best_bins = bins
					best_accuracy = score
					best_classifier = classifier
					best_discretizer = discretizer
	
					print("Currently best accuracy:" + str(best_accuracy))
				
		discretized_dataset_tornado = discretizer.transform(dataset_tornado[:,:-1])
		discretized_dataset_otherweather = discretizer.transform(dataset_otherweather[:,:-1])
		print(discretized_dataset_otherweather.shape)
		# opening and save file for storing discretized dataset
		result_file_dataset_tornado = open(RESULT_DIRECTORY + "tornado/" + FILE_NAME + str(day) + ".csv", "w")
		numpy.savetxt(result_file_dataset_tornado, discretized_dataset_tornado, delimiter=",", comments="")
		result_file_dataset_otherweather = open(RESULT_DIRECTORY + "otherweather/" + FILE_NAME + str(day) + ".csv", "w")
		numpy.savetxt(result_file_dataset_otherweather, discretized_dataset_otherweather, delimiter=",", comments="")
		file_bins = open(RESULT_DIRECTORY + "bins_day"+str(day)+".csv", "w")
		numpy.savetxt(file_bins, numpy.array(best_bins), delimiter=",", comments="")
		if WHOLE_FILE: # Stop iteration if all the days are collapsed in one file
			break
main()
