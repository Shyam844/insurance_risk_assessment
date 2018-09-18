from pandas import read_csv
import numpy

def get_data_from_csv(path):
	# Assumption : The last column in CSV file is the label
	data_frame = read_csv(path)
	data = data_frame.values
	print("Raw Data -> " + str(data.shape))
	rows, cols = data.shape
	x = data[:, :cols-1]	# 'cols-1' is excluded
	y = data[:, cols-1]	# only 'cols-1' column is retrieved
	return x, y

x, y = get_data_from_csv("../dataset/train.csv")
print("x -> " + str(x.shape))
print("y -> " + str(y.shape))

