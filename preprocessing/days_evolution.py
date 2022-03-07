# this code extends the base dataset generating three more instances
# from each instance by rotating each instance matrix

import os
from os.path import isfile
import json
import numpy

DATA_PATH = "../data/featuresExtracted/normalized/quadrants"
FILE_NAME = "quadrants_day"

RESULT_PATH = "../data/daysEvolution/normalized/quadrants"


def daysEvolution():

	# cycling over quadrants folders: quadrants4, quadrants9
	for quadrants_number in (4, 9):

		# cycling over data folders: otherweather, tornado
		for sub_folder in os.listdir(DATA_PATH + str(quadrants_number) + "/"):

			working_directory = DATA_PATH + str(quadrants_number) + "/" + sub_folder + "/"
			result_directory = RESULT_PATH + str(quadrants_number) + "/"

			if not os.path.exists(result_directory):
				os.makedirs(result_directory)

			result_file = open(result_directory + sub_folder + ".csv", "w")

			daily_weather_features_dataset_matrices = []

			daily_weather_features_dataset_headers = []

			# opening files for retrieving variables names and values
			for filename in os.listdir(working_directory):
				daily_weather_features_dataset_file = open(working_directory + filename, "r")
				daily_weather_features_dataset_header = numpy.loadtxt(daily_weather_features_dataset_file, delimiter=",", max_rows=1, dtype=str)
				daily_weather_features_dataset_headers.append(daily_weather_features_dataset_header)
				daily_weather_features_dataset_file = open(working_directory + filename, "r")
				daily_weather_features_dataset_matrix = numpy.loadtxt(daily_weather_features_dataset_file, delimiter=",", skiprows=1)
				daily_weather_features_dataset_matrices.append(daily_weather_features_dataset_matrix)

			# generating output header
			for day in range(0, 4):
				for variable_index, variable_name in enumerate(daily_weather_features_dataset_headers[0]):
					if (variable_index == len(daily_weather_features_dataset_headers[0]) - 1) and (day == 3):
						result_file.write(variable_name + "_" + str(day) + "-" + str(day + 1) + "\n")
					else:
						result_file.write(variable_name + "_" + str(day) + "-" + str(day + 1) + ",")

			# generating output data
			for row in range(0, len(daily_weather_features_dataset_matrices[0])):
				for day in range(0, 4):
					values_difference = daily_weather_features_dataset_matrices[day + 1][row] - daily_weather_features_dataset_matrices[day][row]

					values_difference_string = ','.join(['%f' % value for value in values_difference])

					if (day == 3):
						result_file.write(values_difference_string + "\n")
					else:
						result_file.write(values_difference_string + ",")


			result_file.close()

	return


daysEvolution()
