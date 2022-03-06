import os
from os.path import isfile
import numpy
from sklearn import preprocessing

DATA_PATH = "../data/collections/csv/normal/"
RESULT_PATH = "../data/collections/csv/normalized/"


def featureExtraction():  # if quadrants are 9, the size is 7

	# cycling over data folders: otherweather, tornado
	for folder in os.listdir(DATA_PATH):

		working_directory = DATA_PATH + folder + "/"
		result_directory = RESULT_PATH + folder + "/"

		# cycling over day files: day0, day1, day2, day3, day4
		for filename in os.listdir(working_directory):

			if not os.path.exists(result_directory):
				os.makedirs(result_directory)

			# opening file for storing data normalized
			result_file = open(result_directory + filename, "w")

			# opening file for retrieving variables name
			daily_weather_dataset_file = open(working_directory + filename, "r")
			daily_weather_dataset_header = daily_weather_dataset_file.readline()
			result_file.write(str(daily_weather_dataset_header))

			# opening file for retrieving data
			daily_weather_dataset_file = open(working_directory + filename, "r")
			daily_weather_dataset_data = numpy.loadtxt(daily_weather_dataset_file, delimiter=",", skiprows=1)

			daily_weather_dataset_data_normalized = preprocessing.normalize(daily_weather_dataset_data, axis=0)

			numpy.savetxt(result_file, daily_weather_dataset_data_normalized, delimiter=",", comments="")

			result_file.close()


featureExtraction()
