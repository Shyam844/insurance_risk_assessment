from constants import Constants
from mirror import Mirror
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy

# CONTAINS ALL STATIC MEMBERS
class Pre_processor:

	@staticmethod
	def corr_coeff_analysis(data):
		print("\n=================================================================\n")
		print("Correlation Coefficient Analysis - ")
		pos_up = 1
		pos_down = 0.7
		neg_up = -0.7
		neg_down = -1
		zero_up = 0.001
		zero_down = -0.001
		print("Positively correlated: " + str(pos_down) + " to " + str(pos_up))
		print("Uncorrelated: " + str(zero_down) + " to " + str(zero_up))
		print("Negatively correlated: " + str(neg_down) + " to " + str(neg_up) + "\n")
		corr_coef_grid = numpy.corrcoef(data, rowvar=False)
		rows, cols = corr_coef_grid.shape
		row = 0
		pos_features = []
		neg_features = []
		zero_features = []
		while(row < rows):
			col = 0
			while(col < row):
				corr_coef = corr_coef_grid[row, col]
				if(corr_coef >= pos_down and corr_coef <= pos_up):
					pos_features.append((row+1, col+1))
				elif(corr_coef >= neg_down and corr_coef <= neg_up):
					neg_features.append((row+1, col+1))
				elif(corr_coef >= zero_down and corr_coef <= zero_up):
					zero_features.append((row+1, col+1))
				col = col+1
			row = row+1
		print("Negatively correlated features (" + str(len(neg_features)) + ") : ")
		print(neg_features)
		print("\n=================================================================\n")
		print("Uncorrelated features (" + str(len(zero_features)) + ") : ")
		Pre_processor.get_uncorrelated_dict(zero_features)
		print("\n=================================================================\n")
		print("Positively correlated features (" + str(len(pos_features)) + ") : ")
		print(pos_features)
		print("\n=================================================================\n")
		return neg_features, zero_features, pos_features

	@staticmethod
	def get_uncorrelated_dict(zero_features):
		freq_dict = dict()
		for tuple_ in zero_features:
			first_ = tuple_[0]
			sec_ = tuple_[1]
			if(freq_dict.get(first_) == None):
				freq_dict[first_] = []
			freq_dict[first_].append(sec_)
			if(freq_dict.get(sec_) == None):
				freq_dict[sec_] = []
			freq_dict[sec_].append(first_)
		# print dict
		count = 1
		while(count <= Constants.tot_features):
			print(str(count) + " : ", end="")
			if(freq_dict.get(count) == None):
				print()
			else:
				print(freq_dict[count])
			count = count+1
		return freq_dict

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
