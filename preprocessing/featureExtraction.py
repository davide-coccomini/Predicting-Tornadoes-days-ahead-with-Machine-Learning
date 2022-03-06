import os
import numpy

# DATA_PATH = "../data/collections/csv/normal/"
# RESULT_PATH = "../data/featuresExtracted/normal/quadrants"
DATA_PATH = "../data/collections/csv/normalized/"
RESULT_PATH = "../data/featuresExtracted/normalized/quadrants"

ROWS_PER_CHUNK = 500
CHUNKS_NUMBER = 21


def featureExtraction(quadrants_number=4, quadrants_size=10):  # if quadrants are 9, the size is 7

	# cycling over data folders: otherweather, tornado
	for folder in os.listdir(DATA_PATH):

		working_directory = DATA_PATH + folder + "/"
		result_directory = RESULT_PATH + str(quadrants_number) + "/" + folder + "/"

		# cycling over day files: day0, day1, day2, day3, day4
		for filename in os.listdir(working_directory):

			if not os.path.exists(result_directory):
				os.makedirs(result_directory)

			# opening file for storing features extracted
			result_file = open(result_directory + "quadrants_" + filename, "w")

			# opening file for retrieving variables name
			daily_weather_dataset_file = open(working_directory + filename, "r")
			daily_weather_dataset_header = numpy.loadtxt(daily_weather_dataset_file, delimiter=",", max_rows=1, dtype=str)
			for chunk in range(0, CHUNKS_NUMBER):
				print(str(chunk) + "/" + str(CHUNKS_NUMBER - 1))
				# opening file for retrieving variables data
				daily_weather_dataset_file = open(working_directory + filename, "r")
				try:
					daily_weather_dataset = numpy.loadtxt(daily_weather_dataset_file, delimiter=",", skiprows=((chunk * ROWS_PER_CHUNK) + 1), max_rows=ROWS_PER_CHUNK)
				except:
					print("Too many chunks, skip...")
					break
				if chunk == 0:
					# obtaining header matrix from single line
					daily_weather_dataset_header_matrix = numpy.reshape(daily_weather_dataset_header, (10, 19, 19))

					# generating output header
					for variable_index, variable_matrix in enumerate(daily_weather_dataset_header_matrix):
						variable_name = variable_matrix[0, 0][:-4]
						for quadrant_index in range(0, quadrants_number):
							result_file.write(variable_name + "-" + str(quadrant_index) + "-mean,")
							if (variable_index == 9) and (quadrant_index == (quadrants_number - 1)):
								result_file.write(variable_name + "-" + str(quadrant_index) + "-std\n")
							else:
								result_file.write(variable_name + "-" + str(quadrant_index) + "-std,")

				# generating output features values
				for row in daily_weather_dataset:

					daily_weather_dataset_matrix = numpy.reshape(row, (10, 19, 19))

					for variable_index, variable_matrix in enumerate(daily_weather_dataset_matrix):

						quadrants = numpy.zeros((quadrants_number, quadrants_size, quadrants_size))

						if(quadrants_number == 4):
							quadrants[0] = variable_matrix[0:10, 0:10]
							quadrants[1] = variable_matrix[0:10, 9:19]
							quadrants[2] = variable_matrix[9:19, 0:10]
							quadrants[3] = variable_matrix[9:19, 9:19]

						if(quadrants_number == 9):
							quadrants[0] = variable_matrix[0:7, 0:7]
							quadrants[1] = variable_matrix[0:7, 6:13]
							quadrants[2] = variable_matrix[0:7, 12:19]
							quadrants[3] = variable_matrix[6:13, 0:7]
							quadrants[4] = variable_matrix[6:13, 6:13]
							quadrants[5] = variable_matrix[6:13, 12:19]
							quadrants[6] = variable_matrix[12:19, 0:7]
							quadrants[7] = variable_matrix[12:19, 6:13]
							quadrants[8] = variable_matrix[12:19, 12:19]

						for quadrant_index in range(0, quadrants_number):

							quadrant_row = quadrants[quadrant_index].flatten()

							mean = numpy.mean(quadrant_row)
							std = numpy.std(quadrant_row)

							if (variable_index == 9) and (quadrant_index == (quadrants_number - 1)):
								result_file.write(str(mean) + "," + str(std) + "\n")
							else:
								result_file.write(str(mean) + "," + str(std) + ",")

			result_file.close()


featureExtraction(9, 7)
# featureExtraction(4, 10)
