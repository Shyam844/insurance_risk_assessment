import matplotlib
matplotlib.use('Agg')
from time import time
from pandas import read_csv
import numpy
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from hpelm import ELM

global tot_labels, clf

start_time = time()

def init_global_vars():
	global tot_labels
	tot_labels = 8

def read_csv_(path):
	data_frame = read_csv(path)
	handle_missing_values(data_frame)
	data = data_frame.values
	return data

def handle_missing_values(df):
	df.fillna(df.mean(), inplace=True)

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
	print("Saving Results...")
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

def normalize_data(data):
	scaler = StandardScaler()
	return scaler.fit(data).transform(data)

def get_train_data(path):
	# Assumption : The last column in CSV file is the label
	# Assumption : The 3rd column of CSV file has strings
	print("Getting Train Data...")
	data = read_csv_(path)
	rows, cols = data.shape
	x = data[:, :cols-1]	# 'cols-1' is excluded
	y = data[:, cols-1]	# only 'cols-1' column is retrieved
	handle_str_col(x, 2)
	x = numpy.asarray(x, dtype=numpy.float64)
	y = numpy.asarray(y, dtype=numpy.int)
	print("train_x : " + str(x.shape))
	print("train_y : " + str(y.shape))
	return x, y

def get_test_data(path):
	# Assumption : The 3rd column of CSV file has strings
	print("Getting Test Data...")
	data = read_csv_(path)
	rows, cols = data.shape
	x = data[:, :]
	handle_str_col(x, 2)
	x = numpy.asarray(x, dtype=numpy.float64)
	print("test_x : " + str(x.shape))
	return x

def visualize_pca(x, y):
	print("PCA...")
	pca = PCA(n_components=2)
	x = pca.fit_transform(x)
	print("Post PCA, x -> " + str(x.shape))
	pyplot.scatter(x[:, 0], x[:, 1], c=y, s=9)
	pyplot.savefig("vis_pca.png")

def visualize_tsne(x, y):
	print("T-SNE...")
	tsne_ = TSNE(n_components=2)
	x = tsne_.fit_transform(x)
	print("Post T-SNE, x -> " + str(x.shape))
	pyplot.scatter(x[:, 0], x[:, 1], c=y, s=10)
	pyplot.savefig("vis_tsne.png")

def one_hot_encoding(data):
	# Assumption: data is a 1-D Array
	global tot_labels
	encoded_data = numpy.zeros((len(data), tot_labels))
	ind = 0
	while(ind < len(data)):
		num = data[ind]
		encoded_data[ind][num-1] = 1
		ind = ind+1
	return encoded_data

def one_hot_decoding_partial(data):
	# Assumption: The shape of data should be same as that of "OneHotEncoding"
	rows, cols = data.shape
	row_ind = 0
	decoded_data = numpy.zeros((data.shape))
	while(row_ind < rows):
		row = data[row_ind]
		max_ind = numpy.argmax(row)
		decoded_data[row_ind][max_ind] = 1
		row_ind = row_ind+1
	return decoded_data

def one_hot_decoding_full(data):
	# Assumption: The shape of data should be same as that of "OneHotEncoding"
	# Returns 1-D array
	rows, cols = data.shape
	ind = 0
	decoded_data = []
	while(ind < rows):
		row = data[ind]
		max_ind = numpy.argmax(row)
		decoded_data.append(max_ind+1)
		ind = ind+1
	return decoded_data	

def elm(train_x, train_y, test_x):
	global clf
	print("ELM...")
	global tot_labels
	features = train_x.shape[1]
	train_y = one_hot_encoding(train_y)
	clf = ELM(features, tot_labels)
	clf.add_neurons(220, "sigm")
	clf.train(train_x, train_y, 'CV', 'OP', 'c', k=10)
	pred_y = clf.predict(test_x)
	pred_y = one_hot_decoding_full(pred_y)
	return pred_y

def save_model():
	global clf
	print("Saving Model...")
	f_ = open("model.pkl", "wb")
	pickle.dump(clf, f_)
	f_.close()

def main():
	global clf
	init_global_vars()
	# Get Data
	train_x, train_y = get_train_data("../dataset/train.csv")
	test_x_raw = get_test_data("../dataset/test.csv")

	# Normalize Data
	train_x = normalize_data(train_x)
	test_x = normalize_data(test_x_raw)

	pred_y = elm(train_x, train_y, test_x)
	clf.save("model.pkl")

	# Save Results
	save_results(test_x_raw, pred_y)

'''
	# Visualize Data
	visualize_pca(train_x, train_y)
	visualize_tsne(train_x, train_y)

	# Train the Model
	clf = GaussianNB()
	print("Training...")
	clf.fit(train_x, train_y)

	# Test the Model
	print("Testing...")
	pred_y = clf.predict(test_x)
	
	'''

main()

end_time = time()
elapsed_time = end_time - start_time
print("Processed in " + str(elapsed_time) + " seconds")

