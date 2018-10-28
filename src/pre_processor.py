from constants import Constants
from mirror import Mirror
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy

# CONTAINS ALL STATIC MEMBERS
class Pre_processor:

	@staticmethod
	def get_top_k_features(data, k):
		pca = PCA(n_components=k)
		return pca.fit_transform(data)

	@staticmethod
	def normalize_data(data):
		scaler = StandardScaler()
		return scaler.fit(data).transform(data)

	@staticmethod
	def handle_missing_values(df):
		df.fillna(df.mean(), inplace=True)

	@staticmethod
	def handle_str_col(x, str_ind):
		row = 0
		col = 2
		rows, cols = x.shape
		while(row < rows):
			cur_str = x[row][col]
			x[row][col] = Pre_processor.get_int_from_str(cur_str)
			row = row+1

	@staticmethod
	def get_int_from_str(str_):
		ind = 0
		res = ""
		while(ind < len(str_)):
			char = str_[ind]
			ascii = ord(char)
			res = res + str(ascii)
			ind = ind+1
		return int(res)

	@staticmethod
	def one_hot_encoding(data):
		# Assumption: data is a 1-D Array
		encoded_data = numpy.zeros((len(data), Constants.tot_labels))
		ind = 0
		while(ind < len(data)):
			num = data[ind]
			encoded_data[ind][num-1] = 1
			ind = ind+1
		return encoded_data

	@staticmethod
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

	@staticmethod
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

	@staticmethod
	def feature_mining(train_x, train_y):
		samples, tot_features = train_x.shape
		ind = 0
		while(ind < tot_features):
			path = "./feature_mining/" + str(ind) + ".png"
			Mirror.one_d(train_x[:, ind], train_y, path)
			ind = ind+1
