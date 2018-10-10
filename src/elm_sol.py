from pandas import read_csv
import numpy
from time import time
from hpelm import ELM
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

start_time = time()

def read_csv_(path):
	data_frame = read_csv(path)
	handle_missing_values(data_frame)
	data = data_frame.values
	#print("Raw Data -> " + str(data.shape))
	return data


def get_train_data(path):
	# Assumption : The last column in CSV file is the label
	# Assumption : The 3rd column of CSV file has strings
	data = read_csv_(path)
	rows, cols = data.shape
	x = data[:, :cols-1]	# 'cols-1' is excluded
	y = data[:, cols-1]	# only 'cols-1' column is retrieved
	handle_str_col(x, 2)
	x = numpy.asarray(x, dtype=numpy.float64)
	y = numpy.asarray(y, dtype=numpy.int)
	return x, y

def get_test_data(path):
	# Assumption : The 3rd column of CSV file has strings
	data = read_csv_(path)
	rows, cols = data.shape
	x = data[:, :]
	handle_str_col(x, 2)
	x = numpy.asarray(x, dtype=numpy.float64)
	return x

def handle_missing_values(df):
	df.fillna(df.mean(), inplace=True)

def printToFile(filename, matrix):
	with open(filename, 'w+') as f:
		for row in matrix:
			for j, cols in enumerate(row):
				if(j<len(row)-1):
					f.write(str(cols)+",")
				else:
					f.write(str(cols)+"\n")


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

def save_results(test_x, pred_y):
	test_x = numpy.asarray(test_x, dtype=numpy.int32)
	pred_y = numpy.asarray(pred_y, dtype=numpy.int32)
	col_1 = test_x[:, 0]
	col_2 = pred_y
	book = open("submission.csv", "a")
	book.write("Id,Response\n")
	count = 0
	while(count < len(pred_y)):
		val_1 = str(col_1[count])
		val_2 = str(col_2[count])
		book.write(val_1 + "," + val_2 + "\n")
		count = count+1
	book.close()	

def main():
	# Get Train Data
	print("Getting Data...")
	train_x, train_y = get_train_data("../dataset/train.csv")
	print("train_x -> " + str(train_x.shape))
	print("train_y -> " + str(train_y.shape))

	scaler = StandardScaler()
	train_x = scaler.fit(train_x).transform(train_x)

	# Get Test Data
	# test_x = get_test_data("../dataset/test.csv")
	# print("test_x -> " + str(test_x.shape))

	y_one_hot = numpy.zeros((train_y.shape[0],8))
	for i, row in enumerate(train_y):
		y_one_hot[i,row-1] = 1
	print("y_one_hot -> " + str(y_one_hot.shape))

	
	printToFile("y_one_hot.txt", y_one_hot)
	printToFile("train_x.txt", train_x)
	with open("train_y.txt", 'w+') as f:
		for i in train_y:
			f.write(str(i)+"\n")

	x_train, x_test, y_train, y_test = train_test_split(train_x, y_one_hot, test_size=0.2)
	print(x_train.shape, y_train.shape)
	print(x_test.shape, y_test.shape)

	elm = ELM(x_train.shape[1], y_train.shape[1])
	elm.add_neurons(500, "sigm")
	elm.add_neurons(250, "tanh")
	elm.add_neurons(150, "tanh")
	elm.add_neurons(1000, "sigm")
	elm.add_neurons(500, "tanh")
	elm.add_neurons(250, "sigm")
	elm.add_neurons(50, "tanh")
	# elm.add_neurons(550, "sigm")
	elm.train(x_train, y_train, 'CV', 'OP', 'c', k=10)
	y_predict = elm.predict(x_test)
	y_predicted_one_hot = numpy.zeros((y_predict.shape))
	for i, row in enumerate(y_predict):
		idx = numpy.argmax(row)
		y_predicted_one_hot[i,idx] = 1

	print("accuracy:", accuracy_score(y_test,y_predicted_one_hot))

	

	# Train the Model
	# clf = KNeighborsClassifier(n_neighbors=11)
	# print("Training...")
	# clf.fit(train_x, train_y)

	# # Save the Model
	# print("Saving Model...")
	# joblib.dump(clf, "model.pkl")

	# # Test the Model
	# print("Testing...")
	# pred_y = clf.predict(test_x)
	
	# # Save Results
	# save_results(test_x, pred_y)

main()

end_time = time()
elapsed_time = end_time - start_time
print("Processed in " + str(elapsed_time) + " seconds")

