# this code extends the base dataset generating three more instances
# from each instance by rotating each instance matrix

import os
from os.path import isfile
import json
import numpy

DATA_PATH = "../data/collections/csv/normal/"
RESULT_PATH = "../data/collections/csv/extended/"


def dataExtension():

    # cycling over data folders: otherweather, tornado
    for folder in os.listdir(DATA_PATH):

        working_directory = DATA_PATH + folder + "/"
        result_directory = RESULT_PATH + folder + "/"

        # cycling over day files: day0, day1, day2, day3, day4
        for filename in os.listdir(working_directory):

            if not os.path.exists(result_directory):
                os.makedirs(result_directory)

            # opening file for storing features extracted
            result_file = open(result_directory + filename, "w")

            # opening file for retrieving variables name
            daily_weather_dataset_file = open(working_directory + filename, "r")
            daily_weather_dataset_header = numpy.loadtxt(daily_weather_dataset_file, delimiter=",", max_rows=1, dtype=str)

            # opening file for retrieving variables data
            daily_weather_dataset_file = open(working_directory + filename, "r")
            daily_weather_dataset = numpy.loadtxt(daily_weather_dataset_file, delimiter=",", skiprows=1)

            # copying header
            daily_weather_dataset_header_string = ','.join(['%s' % string for string in daily_weather_dataset_header])
            result_file.write(daily_weather_dataset_header_string + "\n")

            # generating new instances
            for row in daily_weather_dataset:

                daily_weather_dataset_matrix = numpy.reshape(row, (10, 19, 19))

                for rotations_number in range(0, 4):

                    for variable_index, variable_matrix in enumerate(daily_weather_dataset_matrix):

                        rotated_variable_matrix = numpy.rot90(variable_matrix, rotations_number)

                        rotated_variable_matrix_row = rotated_variable_matrix.flatten()

                        rotated_variable_matrix_row_string = ','.join(['%f' % value for value in rotated_variable_matrix_row])

                        if (variable_index == 9):
                            result_file.write(rotated_variable_matrix_row_string + "\n")
                        else:
                            result_file.write(rotated_variable_matrix_row_string + ",")


            result_file.close()


dataExtension()
