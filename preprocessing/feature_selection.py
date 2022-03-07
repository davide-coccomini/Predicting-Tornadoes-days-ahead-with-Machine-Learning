import os
import numpy

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest,chi2,f_classif 
from sklearn.metrics import normalized_mutual_info_score


QUADRANTS_PATH = "../data/featuresExtracted/normal/quadrants"
FILE_NAME = "quadrants_day"

RESULT_PATH = "../data/featuresSelected/quadrants"


TRAINING_SET_PERCENTAGE = 85
GOAL_FEATURES_NUMBER = {4:70, 9:140}
def main():

	for quadrants_number in (4,9):
		for day in range(0,6):
			tornado_path = QUADRANTS_PATH + str(quadrants_number) + "/tornado/"
			otherweather_path = QUADRANTS_PATH + str(quadrants_number) + "/otherweather/"

			tornado_features_file = open(tornado_path + FILE_NAME + str(day) + ".csv", "r")
			otherweather_features_file = open(otherweather_path + FILE_NAME + str(day) + ".csv", "r")

			# Read features
			dataset_tornado = numpy.loadtxt(tornado_features_file, delimiter=",", skiprows=1)
			dataset_otherweather = numpy.loadtxt(otherweather_features_file, delimiter=",", skiprows=1)

			tornado_features_file = open(tornado_path + FILE_NAME + str(day) + ".csv", "r")
			otherweather_features_file = open(otherweather_path + FILE_NAME + str(day) + ".csv", "r")

			dataset_tornado_header = numpy.loadtxt(tornado_features_file, delimiter=",", max_rows=1, dtype=str)
			dataset_otherweather_header = numpy.loadtxt(otherweather_features_file, delimiter=",", max_rows=1, dtype=str)

			# Labelling
			tornado_labels = [[1] for i in range(0, len(dataset_tornado))]
			otherweather_labels = [[0] for i in range(0, len(dataset_otherweather))]

			dataset_tornado = numpy.append(dataset_tornado, tornado_labels, axis=1)
			dataset_otherweather = numpy.append(dataset_otherweather, otherweather_labels, axis=1)
			complete_dataset = numpy.append(dataset_tornado, dataset_otherweather,axis=0)
			
			# Split in values and labels
			tornados_number = len(dataset_tornado)
			dataset_values = complete_dataset[:,:-1]
			dataset_labels = (complete_dataset[:,-1]).astype(int)
			

			print("Feature selection ...")
			plt.figure(1)
			plt.clf()
			selector = SelectKBest(f_classif , k=GOAL_FEATURES_NUMBER[quadrants_number])
			selector.fit(dataset_values, dataset_labels)
			scores = -numpy.log10(selector.pvalues_)
			scores /= scores.max()
			print(len(complete_dataset[0]))
			print([i for i in range(0, len(complete_dataset[0]))])
			plt.bar([i for i in range(0,len(complete_dataset[0])-1)], scores, width=.2,
				label=r'Univariate score ($-Log(p_{value})$)', color='darkorange',
				edgecolor='black')
			plt.show()

			dataset_tornado_reduced = selector.fit_transform(dataset_values[:tornados_number,:], dataset_labels[:tornados_number])
			dataset_tornado_header_reduced = dataset_tornado_header[selector.get_support()]
			dataset_otherweather_reduced = selector.fit_transform(dataset_values[tornados_number:,:], dataset_labels[tornados_number:])
			dataset_otherweather_header_reduced = dataset_otherweather_header[selector.get_support()]


			print("New shape tornados: "+str(dataset_tornado_reduced.shape))
			print("New shape otherweathers: "+str(dataset_otherweather_reduced.shape))

			tornado_result_directory = RESULT_PATH + str(quadrants_number) + "/tornado/"
			otherweather_result_directory = RESULT_PATH + str(quadrants_number) + "/otherweather/"

			if not os.path.exists(tornado_result_directory):
   				os.makedirs(tornado_result_directory)
			
			if not os.path.exists(otherweather_result_directory):
   				os.makedirs(otherweather_result_directory)

			tornado_result_file = open(tornado_result_directory + "quadrants_day" + str(day) + ".csv", "w")
			otherweather_result_file = open(otherweather_result_directory + "quadrants_day" + str(day) + ".csv", "w")

			dataset_tornado_header_reduced_string =	','.join(['%s' % string for string in dataset_tornado_header_reduced])
			dataset_otherweather_header_reduced_string = ','.join(['%s' % string for string in dataset_otherweather_header_reduced])


			numpy.savetxt(tornado_result_file, dataset_tornado_reduced, delimiter=",", header=dataset_tornado_header_reduced_string, comments="")
			numpy.savetxt(otherweather_result_file, dataset_otherweather_reduced, delimiter=",", header=dataset_otherweather_header_reduced_string, comments="")


main()
