import os
from os.path import isfile
import json
import numpy

#matrix = numpy.zeros((19, 19), dtype=float)
DATA_PATH = "../data/collections/json/"
RESULT_PATH = "../data/collections/csv/normal/"


def main(days_range=[0, 5]):

    for filename in os.listdir(DATA_PATH):

        result_directory = RESULT_PATH + filename[:-5] + "/"

        if not os.path.exists(result_directory):
            os.makedirs(result_directory)

        day_files = []

        for i in range(days_range[0], days_range[1]):
            day_files.append(open(result_directory + "day" + str(i) + ".csv", "w"))

        data_file = open(DATA_PATH + filename, "r")
        data_lines = data_file.readlines()

        weather = json.loads(data_lines[0])
        variables = weather["data"]
        for variable_index, variable in enumerate(variables):
            for day_index, day_values_matrix in enumerate(variables[variable]):
                for lat_index in range(0, 19):
                    for lon_index in range(0, 19):
                        if (variable_index == (len(variables) - 1)) and (lat_index == 18) and (lon_index == 18):
                            day_files[day_index].write(
                                variable + "-" + str(lat_index) + "-" + str(lon_index) + "\n")
                        else:
                            day_files[day_index].write(
                                variable + "-" + str(lat_index) + "-" + str(lon_index) + ",")

        for data_line in data_lines:
            weather = json.loads(data_line)
            variables = weather["data"]

            for variable_index, variable in enumerate(variables):

                for day_index, day_values_matrix in enumerate(variables[variable]):

                    matrix = numpy.array(day_values_matrix)

                    for lat_index in range(0, 19):
                        for lon_index in range(0, 19):
                            if (variable_index == (len(variables) - 1)) and (lat_index == 18) and (lon_index == 18):
                                day_files[day_index].write(str(matrix[lat_index][lon_index]) + "\n")
                            else:
                                day_files[day_index].write(str(matrix[lat_index][lon_index]) + ",")



        data_file.close()
        for i in range(days_range[0], days_range[1]):
            day_files[i].close()


main()
