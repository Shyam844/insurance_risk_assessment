import matplotlib
matplotlib.use('Agg')
from pandas import read_csv
import numpy
from time import time
from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot


start_time = time()

def normalize_data(data):
	scaler = StandardScaler()
	return scaler.fit(data).transform(data)

def read_csv_(path):
	data_frame = read_csv(path)
	handle_missing_values(data_frame)
	data = data_frame.values
	print("Raw Data -> " + str(data.shape))
	return data


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

	
def main():
	# Get Data
	train_x, train_y = get_train_data("../dataset/train.csv")
	test_x = get_test_data("../dataset/test.csv")

	# Normalize Data
	train_x = normalize_data(train_x)
	test_x = normalize_data(test_x)

	# Visualize Data
	visualize_pca(train_x, train_y)
	visualize_tsne(train_x, train_y)
'''
	# Train the Model
	clf = GaussianNB()
	print("Training...")
	clf.fit(train_x, train_y)

	# Save the Model
	print("Saving Model...")
	joblib.dump(clf, "model.pkl")

	# Test the Model
	print("Testing...")
	pred_y = clf.predict(test_x)
	
	# Save Results
	save_results(test_x, pred_y)'''

main()

end_time = time()
elapsed_time = end_time - start_time
print("Processed in " + str(elapsed_time) + " seconds")

