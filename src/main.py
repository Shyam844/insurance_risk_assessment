from pandas import read_csv
import numpy
from time import time
from sklearn.naive_bayes import GaussianNB
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score

start_time = time()


def get_data_from_csv(path):
	# Assumption : The last column in CSV file is the label
	# Assumption : The 3rd column of CSV file has strings
	data_frame = read_csv(path)
	data = data_frame.values
	print("Raw Data -> " + str(data.shape))
	rows, cols = data.shape
	x = data[:, :cols-1]	# 'cols-1' is excluded
	y = data[:, cols-1]	# only 'cols-1' column is retrieved
	handle_str_col(x, 2)
	#print(x[2, 29]) (Missing Values in data)
	return x, y

def handle_str_col(x, str_ind):
	row = 0
	col = 2
	rows, cols = x.shape
	while(row < rows):
		cur_str = x[row][col]
		x[row][col] = get_int_from_str(cur_str)
		row = row+1

def get_int_from_str(str_):
	ind = 0
	res = ""
	while(ind < len(str_)):
		char = str_[ind]
		ascii = ord(char)
		res = res + str(ascii)
		ind = ind+1
	return int(res)


def main():
	# Get Train Data
	train_x, train_y = get_data_from_csv("../dataset/train.csv")
	print("train_x -> " + str(train_x.shape))
	print("train_y -> " + str(train_y.shape))

	# Get Test Data
	test_x, test_y = get_data_from_csv("../dataset/test.csv")
	print("test_x -> " + str(test_x.shape))
	print("test_y -> " + str(test_y.shape))

	# Train the Model
	clf = GaussianNB()
	print("Training...")
	clf.fit(train_x, train_y)

	# Save the Model
	print("Saving Model...")
	joblib.dump(clf, "model.pkl")

	# Test the Model
	print("Testing...")
	y_pred = clf.predict(x_test)

	# Calculate Accuracy
	print("Calculating Accuracy...")
	score = accuracy_score(y_test, y_pred)
	print("Accuracy -> " + str(score))
	

main()

end_time = time()
elapsed_time = end_time - start_time
print("Processed in " + str(elapsed_time) + " seconds")

