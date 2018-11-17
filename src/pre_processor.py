from constants import Constants
from mirror import Mirror
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

# CONTAINS ALL STATIC MEMBERS
class Pre_processor:

	@staticmethod
	def reduce_features(data, features_to_consider, features_to_ignore, one_indexing=False, print_str="data"):
		if(len(features_to_ignore) != 0):
			features_to_ignore = Pre_processor.one_to_zero_index(features_to_ignore)
			# init feature array; 1D array of length Constants.tot_features; each value is either 0 or 1; 1 represents 'consider the feature'.
			feature_array = []
			ind = 0
			while(ind < len(data[0])):
				feature_array.append(1)
				ind = ind+1
			# update feature array with features_to_ignore array
			ind = 0
			while(ind < len(features_to_ignore)):
				num = features_to_ignore[ind]
				feature_array[num] = 0
				ind = ind+1
			ind = 0
			# update features_to_consider array
			while(ind < len(feature_array)):
				num = feature_array[ind]
				if(num == 1):
					features_to_consider.append(ind)
				ind = ind+1
		elif(len(features_to_consider) != 0):
			if(one_indexing == True):
				features_to_consider = Pre_processor.one_to_zero_index(features_to_consider)
		data = data[:, features_to_consider]
		print("Post feature reduction, " + print_str + ": " + str(data.shape))
		return data

	@staticmethod
	def one_to_zero_index(arr):
		ind = 0
		while(ind < len(arr)):
			arr[ind] = arr[ind]-1
			ind = ind+1
		return arr

	@staticmethod
	def recursive_feature_elimination(train_x, train_y):
		clf = LogisticRegression()
		rfe = RFE(clf, 3)
		print("\n==================================")
		print("Recursive Feature Elimination...")
		rfe = rfe.fit(train_x, train_y)
		print(rfe.support_)
		print("Ranking:")
		print(rfe.ranking_)
		print("==================================\n")


	@staticmethod
	def random_forest_feature_engineering(train_x, train_y):
		rand_forest = ExtraTreesClassifier(n_estimators=128, n_jobs=30)
		print("\nFitting Random Forest...")
		rand_forest.fit(train_x, train_y)
		feature_imps = rand_forest.feature_importances_.tolist()
		# Add to dict
		ind = 0
		feature_imps_dict = dict()
		while(ind < len(feature_imps)):
			feature_imps_dict[ind+1] = feature_imps[ind]
			ind = ind+1
		sorted_keys = sorted(feature_imps_dict, key=feature_imps_dict.get, reverse=True)
		print("\n===================================================")
		print("feature: importance")
		for key_ in sorted_keys:
			print(str(key_) + ": " + str(feature_imps_dict[key_]))
		print("===========================================\n")

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
			print(str(count) + " : ")
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
